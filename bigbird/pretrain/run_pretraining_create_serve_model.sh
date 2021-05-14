#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
export PYTHONPATH="/app/ko_bigbird:$PYTHONPATH"
cd /app/ko_bigbird/bigbird/pretrain/

/usr/local/bin/poetry run python ./run_pretraining.py \
  --init_checkpoint=../model/pretrained/ \
  --data_dir=../datasource/tf_pretrained_data \
  --output_dir=../model/serving \
  --vocab_model_file=../bpe_model/ko_big_bird.model \
  --max_encoder_length=1024 \
  --max_predictions_per_seq=75 \
  --masked_lm_prob=0.15 \
  --do_train=false \
  --do_eval=false \
  --do_export=true \
  --train_batch_size=1 \
  --eval_batch_size=1 \
  --batch_size=1 \
  --preprocessed_data=true

/usr/local/bin/poetry run python ./run_serve_test.py \
  --serve_model=../model/serving/ \
  --vocab_dir=../bpe_model/ko_big_bird.model \
  --test_text="비트코인의 핵심은 블록체인 기술이다"