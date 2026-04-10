#!/usr/bin/env bash
set -euo pipefail

echo "Converting Llava Single Turn dataset to custom format..."
python3 -m  transfer.custom_conversion \
--in "../llava_instruct/llava_instruct/chat_single_turn.jsonl" \
--out "../llava_instruct/custom_llava/llava_single.jsonl" \
--src-name "llava_single"

echo "Converting Llava Multi Turn dataset to custom format..."
python3 -m  transfer.custom_conversion \
--in "../llava_instruct/llava_instruct/chat_multi_turn.jsonl" \
--out "../llava_instruct/custom_llava/llava_multi.jsonl" \
--src-name "llava_multi"

echo "Converting GQA dataset to custom format..."
python3 -m transfer.custom_conversion \
--in "../DataOptim/gqa.jsonl" \
--out "../DataOptim/custom_data_optim/gqa.jsonl" \
--src-name "gqa"

echo "Converting Ref3Rec dataset to custom format..."
python3 -m transfer.custom_conversion \
--in "../DataOptim/ref3rec.jsonl" \
--out "../DataOptim/custom_data_optim/ref3rec.jsonl" \
--src-name "ref3rec"

echo "Converting Ref3Reg dataset to custom format..."
python3 -m transfer.custom_conversion \
--in "../DataOptim/ref3reg.jsonl" \
--out "../DataOptim/custom_data_optim/ref3reg.jsonl" \
--src-name "ref3reg"

echo "Converting Shikra dataset to custom format..."
python3 -m transfer.custom_conversion \
--in "../DataOptim/shikra.jsonl" \
--out "../DataOptim/custom_data_optim/shikra.jsonl" \
--src-name "shikra"

echo "Converting referring_qa dataset to custom format..."
python3 -m transfer.svit_convert \
  --in ../SVIT/referring_qa.json \
  --out ../SVIT/custom_svit/svit_referring_qa.jsonl \
  --type referring_qa \
  --img_tmpl "{:07d}.jpg"

echo "Converting detail_description dataset to custom format..."
python3 -m transfer.svit_convert \
  --in ../SVIT/detail_description.json \
  --out ../SVIT/custom_svit/svit_detail.jsonl \
  --type detail_description \
  --img_tmpl "{:07d}.jpg"

echo "Converting conversation dataset to custom format..."
python3 -m transfer.svit_convert \
  --in ../SVIT/conversation.json \
  --out ../SVIT/custom_svit/svit_conversation.jsonl \
  --type conversation \
  --img_tmpl "{:07d}.jpg"

echo "Converting complex_reasoning dataset to custom format..."
python3 -m  transfer.svit_convert \
  --in ../SVIT/complex_reasoning.json \
  --out ../SVIT/custom_svit/svit_complex.jsonl \
  --type complex_reasoning \
  --img_tmpl "{:07d}.jpg"



echo "Conversion completed."