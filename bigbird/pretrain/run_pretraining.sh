#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
export PYTHONPATH="/app/ko_bigbird:$PYTHONPATH"

cd /app/ko_bigbird/bigbird/pretrain/

/usr/local/bin/poetry run python ./run_pretraining.py \
  --init_checkpoint=../model/pretrained/ \
  --data_dir=../datasource/tf_pretrained_data \
  --output_dir=../model/pretrained \
  --vocab_model_file=../bpe_model/ko_big_bird.model \
  --num_train_steps=1000000000 \
  --max_encoder_length=1024 \
  --max_predictions_per_seq=75 \
  --masked_lm_prob=0.15 \
  --do_train=true \
  --do_eval=false \
  --do_export=false \
  --train_batch_size=7 \
  --batch_size=7 \
  --preprocessed_data=true
