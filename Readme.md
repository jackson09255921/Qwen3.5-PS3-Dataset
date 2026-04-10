Special Token: 
影像 / 區域：<image>、<bbox>
語詞區域標記：<st>、<ed>

[Module 定義]（加上「強化 VQA / caption」觀點）
Vision anchor（PS3 selection）
目標：強化 vision encoder 在 VQA / caption 時「看對區域」的能力。
標籤：gt_selection_maps
由 bboxes + PS3 feature map 尺寸計算 → Gaussian heatmap。
作為 top‑down prompt 選圖時的 supervision，學「問題提到的目標在圖上哪裡」。
​
Language anchor（cross‑attention）
目標：在已看對區域的前提下，精細對齊「phrase ↔ 影像區域」，讓模型知道每段描述在指哪個實體。
標籤：gt_vision_masks
由 (t_start, t_end, bbox_idx) + 影像 token layout → text×image 的 token span mask。
用於 cross‑attention 的對齊 loss，強化「<st>phrase<ed> <bbox>」對應的 image tokens。


Input Prompt（不改，只補一句說明）
單一圖片多目標：

<image>\n + 文字（可含 <st>/<ed>/<bbox>）
或 文字\n<image>（保留 LLaVA 的雙 pattern）。
多輪對話：
只在第一輪 Q 放一次 <image> 即可，共享同一張圖 context。

說明：訓練時會在文字中加上 <st>/<ed>/<bbox> 給 language anchor 用，但 benchmark / 真實推論 可以只用自然語言問句（不帶這些 supervision tag）。


------------------------------------------------------------------------------------------------------------
Conversations 範例:

頂層 schema（統一版，含 image_index）
{
  "id": "sample_0001",
  "task_type": "ground_rec",          // ground_rec / ground_region_qa / ground_region_caption / vqa / caption / text_instruct / multi_image_*
  "images": ["0001_a.jpg", "0001_b.jpg"],
  "meta": { ... },
  "bboxes": [
    { "image_index": 0, "bbox": [0.147, 0.022, 0.258, 0.325] },
    { "image_index": 1, "bbox": [0.526, 0.152, 0.625, 0.717] }
  ],
  "grounding_targets": [
    {
      "type": "refer",                // refer / region_qa / region_caption
      "phrase": "the man in red",
      "bbox_index": 0                 // 對應上面 bboxes 的 index（anchor 用）
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n... text with <st>...<ed> <bbox> ..."
    },
    {
      "from": "assistant",
      "value": "..."
    }
  ]
}

1. Grounding‑REC（問問題找框）
{
  "id": "ground_rec_0001",
  "task_type": "ground_rec",
  "images": ["scene.jpg"],
  "bboxes": [
    { "image_index": 0, "bbox": [0.20, 0.30, 0.40, 0.70] }   // idx 0: man in red
  ],
  "grounding_targets": [
    {
      "type": "refer",
      "phrase": "the man in red",
      "bbox_index": 0
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nWhere is the <st>man in red<ed> <bbox> located in the image?"
    },
    {
      "from": "assistant",
      "value": "The <st>man in red<ed> <bbox> is standing on the left side."
    }
  ]
}

2. Grounding‑Region QA（給框問問題）
{
  "id": "ground_region_qa_0001",
  "task_type": "ground_region_qa",
  "images": ["scene.jpg"],
  "bboxes": [
    { "image_index": 0, "bbox": [0.20, 0.30, 0.40, 0.70] }   // idx 0: region to ask about
  ],
  "grounding_targets": [
    {
      "type": "region_qa",
      "bbox_index": 0
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nGiven the highlighted region <bbox>, what is this object?"
    },
    {
      "from": "assistant",
      "value": "It is a <st>man<ed> <bbox>."
    }
  ]
}

3. Grounding‑Region Caption（給框產 caption）
{
  "id": "ground_region_caption_0001",
  "task_type": "ground_region_caption",
  "images": ["scene.jpg"],
  "bboxes": [
    { "image_index": 0, "bbox": [0.20, 0.30, 0.40, 0.70] }
  ],
  "grounding_targets": [
    {
      "type": "region_caption",
      "bbox_index": 0,
      "caption": "a man in a red shirt holding a ball"
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nDescribe in detail the object inside the <bbox> region."
    },
    {
      "from": "assistant",
      "value": "It is <st>a man in a red shirt holding a ball<ed> <bbox>."
    }
  ]
}

----------------------------------------同時 serve VQA/caption 強化（靠 anchor）和未來的 grounding fine‑tune-----------------------------------------------------

4. VQA（單圖）
{
  "id": "vqa_0001",
  "task_type": "vqa",
  "images": ["vqa.jpg"],
  "bboxes": [],
  "grounding_targets": [],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nWhat is the main object in this image?"
    },
    {
      "from": "assistant",
      "value": "A dog is sitting on the grass."
    }
  ]
}

5. Image captioning
{
  "id": "caption_0001",
  "task_type": "caption",
  "images": ["coco_0001.jpg"],
  "bboxes": [],
  "grounding_targets": [],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nPlease describe this image."
    },
    {
      "from": "assistant",
      "value": "A group of people are walking across a busy city street."
    }
  ]
}

6. 純文字 text‑instruct
{
  "id": "text_instruct_0001",
  "task_type": "text_instruct",
  "images": [],
  "bboxes": [],
  "grounding_targets": [],
  "conversations": [
    {
      "from": "user",
      "value": "Explain what LoRA is in 3 sentences."
    },
    {
      "from": "assistant",
      "value": "LoRA is ..."
    }
  ]
}

7. Bonus：多圖 + image_index（multi_image_vqa 範例）
{
  "id": "multi_image_vqa_0001",
  "task_type": "multi_image_vqa",
  "images": ["before.jpg", "after.jpg"],
  "bboxes": [
    { "image_index": 0, "bbox": [0.10, 0.50, 0.30, 0.80] },   // car in first image
    { "image_index": 1, "bbox": [0.00, 0.00, 0.00, 0.00] }    // dummy / other region
  ],
  "grounding_targets": [
    {
      "type": "refer",
      "phrase": "the car that moved away",
      "bbox_index": 0
    }
  ],
  "conversations": [
    {
      "from": "user",
      "value": "<image>\nCompare these images. What changed between the first and second image?"
    },
    {
      "from": "assistant",
      "value": "In the second image, the <st>car<ed> <bbox> has moved away from the parking spot."
    }
  ]
}


----------------------------------------------------------------------------------------------------------------

摘要
A：對齊多模態指令（變成穩定 VLM）

B1：強化一般 VQA / caption / multi‑turn

B2：開 Vision / Language anchor，用 bbox 強化「看對地方、說對對象」

C：純文字補 LLM

D：在 grounded base‑model 上，透過離散座標 SFT 學輸入輸出 bbox，專門打 grounding benchmark（RefCOCO / GUI）。

-----------------------------------------------------------------------------------------------
[新 LoRA 主力集 ]

A. Align LoRA（Visual Instruction Align, 打底成穩定 VLM）[40k]
    1. [20k（50%）] liuhaotian/LLaVA-Instruct-150K（conversation size = 1，單輪簡單 QA / 描述）
    2. [20k（50%）] BAAI/SVIT（單輪對話 + simple QA）

    目的：先把小模型對「<image> + 指令」的基本格式、回答風格對齊到 LLaVA / SVIT 分佈，打好視覺指令基礎。

    PS3：freeze
    Projector：on（主角，只校正圖像特徵到 LLM 空間）
    Vision‑LoRA：off
    LLM‑LoRA：off
    after_image：on
    after_prompt：off
    tune_top_down_selection：off

--------------------------------------------------------------------------------------------------
B. SFT‑VLM LoRA（視覺任務 SFT：先強化一般 VQA / caption，再加 grounding anchor）[40k + 20k]

B1. 一般視覺 SFT（complex VQA / caption / multi‑turn）[40k]
    1. [20k（50%）] liuhaotian/LLaVA-Instruct-150K（conversation size > 1，hard / multi‑turn）
    2. [20k（50%）] BAAI/SVIT（complex reasoning, referring QA）

    目的：在 Align 之上，讓模型學會多輪對話、較難的 VQA / caption / referring QA，不啟用 bbox anchor，只當一般 VLM SFT。

    PS3：freeze
    Projector：on（LR 比 Stage‑A 再小一階，微調對齊）
    Vision‑LoRA：off
    LLM‑LoRA：on
    after_image：off
    after_prompt：on
    tune_top_down_selection：off

--------------------------------------------------------------------------------------------------
B2. Grounding 強化（Vision / Language anchor，上 bbox 當「看對地方、說對對象」的強化劑）[20k]
    3. [20k] BAAI/DataOptim {
           [2k（10%）] Shikra-RD
           [6k（30%）] GQA
           [12k（60%）] RefCOCO/g/+（Ref3Rec + Ref3Reg）
       }

    目的：
        - Vision anchor：用 bbox → gt_selection_maps，訓練 top‑down PS3 selection，強化「問題提到的目標時要看哪裡」。
        - Language anchor：用 <st> / <ed> / <bbox> + grounding_targets，把 phrase ↔ region 對齊，強化「描述對象是誰 / 在哪裡」。
        - 整體上把小模型從一般 VLM 拉成 grounded VLM，讓 VQA / caption 更看圖、更對目標，並為之後離散座標 fine‑tune 打底。

    PS3：freeze
    Projector：on（依然更新）
    Vision‑LoRA：on
    LLM‑LoRA：on（學 mapping「語言指令 → 座標 / region 描述」，同時強化 VQA / caption 的 grounding）
    after_image：off
    after_prompt：on
    tune_top_down_selection：on

--------------------------------------------------------------------------------------------------
C. SFT‑Text LoRA（純文字：在不破壞多模態的前提下補 LLM 能力）[60k]
    1. [10k] databricks/databricks-dolly-15k
    2. [20k] xiaozheyao/megachat , [Nk] allenai/tulu-3-sft-personas-instruction-following
    3-1. [5.6k（40%）] allenai/math_qa, [2.8k（20%）] HuggingFaceH4/MATH-500
    3-2. [6k（100%）] HuggingFaceH4/CodeAlpaca_20K

    目的：只在純文字上調整 LLM（instruction / reasoning / code / math），檢查 text‑only benchmark，同時確認不明顯 hurt 多模態。

    PS3：freeze
    Projector：freeze
    Vision‑LoRA：off
    LLM‑LoRA：on（這個 Stage 專門調 LLM 部分）
    after_image：off
    after_prompt：on
    tune_top_down_selection：off
--------------------------------------------------------------------------------------------------
D. Grounding‑Coord LoRA（離散座標 / 定位 SFT：學輸入輸出座標）[N k, 視資料而定]

    資料來源（示意）：
        1. RefCOCO / RefCOCO+ / RefCOCOg（Ref3Rec / Ref3Reg）
        2. Shikra‑RD / GQA 中有 bbox 的子集
        3. 自建 GUI / ScreenSpot 類螢幕 grounding 資料（若要做 UI agent）

    資料形式：
        - 保留原有 schema 的 bboxes / grounding_targets（給 Vision / Language anchor 用）。
        - 額外在 conversations 中，用離散座標 token 表示 bbox，例如：
          User:   "<image>\nPlease localize the man in red."
          Assist: "bbox0(<x12><y05><x40><y60>)\nHe is standing on the left side of the image."
        - 或：輸入也可以包含離散座標（支援 bbox+text→text / bbox+text→bbox+text 等互動）。

    目的：
        - 在已經經過 Stage‑B2 強化的 grounded VLM 上：
            * 教模型「用離散 <xNN>/<yNN> token 寫出 bbox」，對齊 RefCOCO 類 IoU benchmark。
            * 若需要，也讓模型能讀懂離散座標作為條件（bbox+text→text / bbox+text→bbox+text）。
        - 讓模型在保持 VQA / caption 能力的前提下，在 grounding 任務（RefCOCO / ScreenSpot）上達到 SOTA 級表現。

    設定建議：
        PS3：freeze（視覺已在 B2 打好 grounding，不再動）
        Projector：視狀況小 LR 或 freeze（通常以穩為主）
        Vision‑LoRA：off（不再改 vision anchor，本階段重點在「把已有定位能力顯性化成座標輸出」）
        LLM‑LoRA：on（學習離散座標 I/O + output 格式）
        after_image：off
        after_prompt：on（仍用自然語言 prompt + 座標輸出）
        tune_top_down_selection：off（避免在最後階段再改 selection 行為）

    備註：
        - 若想在這階段繼續利用 anchor，可選擇少量保留 B2 那種 <st>/<ed>/<bbox> + grounding_targets 的樣本，
          但主要 loss 放在「座標 token + 文字」的 next‑token SFT 與 IoU‑based model selection 上。

-----------------------------------------------------------------------------------------------------------


[「大補帖 / optional pool」，只在需要時抽少量樣本補某些任務。5k–10k]
A .Align LoRA（Visual Instruction Align）[20k]
    1. CC3M（LLaVA‑CC3M‑Pretrain‑595K）
    2. COYO‑700M（子集）
    3. MMC4‑core
    4. WIT（Wikipedia Image Text）

B. SFT‑VLM LoRA（視覺任務 SFT）
    1. VisualGenome                         [5k（region caption / QA）]
    2. VCR（Visual Commonsense Reasoning）  [5k（commonsense QA）]
    3. ScienceQA                            [5k（如果 benchmark 弱再加）]
    4. Shot2Story                           [5k（video→story）]
    5. YouCook2                             [5k（video caption）]
    6. ShareGPT‑Video                       [5k（video instruction）]                  
    # （ScienceQA / video‑QA / commonsense）表現弱

C. SFT‑Text LoRA（純文字）
    1. Open‑Orca/FLAN 子集 (可選)           [5k–10k（如果你想更貼近 VILA 的 text 配方）]
    2. GSM8K‑ScRel                          [5k–10k（如果 math benchmark 還是不夠）]


--------------------------------------------------------------------------------------------------  
Benchmarks
1. 一般 / reasoning：
    MMMU (val)、MathVista (testmini)、MMStar (val)

2. 圖文 / OCR / doc：
    TextVQA (val)、DocVQA (test/val)、AI2D（圖表/科學圖）、ChartQA / InfoVQA（若跑得動）
    4K / 高解析（專門看 PS3 優勢）：、4KPro（PS3 論文主打的 4K QA benchmark）

3. Grounding / referring：
    RefCOCO / RefCOCO+ / RefCOCOg（指向式定位）
    GQA（結構化視覺推理 + 關係）

--------------------------------------------------------------------------------------------------  
Ablations:
1. Stage‑A Align
    目的：看「Align 40k 配方」對比「只用其中一個源」。
    Ablation
        A0：不做 Align，直接從 base checkpoint 進 B。
        A1：Align 只用 LLaVA simple。
        A2：Align 只用 SVIT simple。
        A3：現在這版 LLaVA:SVIT = 1:1（主線）。
    測：
        VILA 那圈 image QA benchmark 的 easy 子集（例如 VQAv2 / TextVQA / DocVQA）
        看 A0 vs A3 的差距，主要是「格式穩定度 / hallucination / basic VQA accuracy」。

2. Stage‑B1 VLM（hard / complex）
    目的：比較不同 vision/LLM 組合、以及「有無 B1」對整體視覺能力的影響。
    Ablation
        B1‑off：跳過 B1，只跑 B2+Stage‑C。
        B1‑full（主線）：先 B1 再 B2。
    測：
        MMMU, MathVista, MMStar, AI2D, ScienceQA 等需要 reasoning 的視覺 benchmark。
        看 B1‑off vs full 在 reasoning 任務上的差距。

3. Stage‑B2 Grounding
    目的：專門看 grounding 配方有效性，以及 vision‑LoRA / top‑down selection 的貢獻。
    Ablation
        G0：不做 B2（只有 B1 + C）。
        G1：B2 只開 projector+LLM‑LoRA，Vision‑LoRA & tune_top_down_selection=off。
        G2（主線）：B2 開 Vision‑LoRA + tune_top_down_selection=on。
    測：
        RefCOCO/g/+、GQA、你額外設計的 GUI / UI grounding 任務（可以用 4KPro 中的 UI 類子集）。
        再加 1–2 個 VILA 高解析 benchmark（例如 4KPro / RealWorldQA）。   ​
    小模型對比
        同樣跑在 SmolVLM2‑500M / PS3‑1.5K‑small LLM 上，證明你 recipe 在小模型也有效。

4. Stage‑C Text
    目的：看 Text‑SFT 對多模態能力有沒有破壞，順便量測 text benchmark。
    Ablation
        C0：不做 C，只跑 A+B。
        C1（主線）：做 C 全套。
    測：
        Text‑only：例如 MMLU / GPT‑4 MT‑Bench /一般 chat 評測（你可以跑 fastchat/MT‑bench 或 arena‑style 評分）。
        漏跑多模態 regression：挑幾個代表性的 VQA benchmark，再測一次，看 C 是否明顯 hurt VQA。

