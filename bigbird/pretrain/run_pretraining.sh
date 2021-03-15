#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2

export PYTHONPATH='/app/ko_bigbird/'

poetry run python ./run_pretraining.py \
  --data_dir=../datasource/tf_pretrained_data \
  --output_dir=../model/pretrained \
  --vocab_model_file=../bpe_model/ko_big_bird.model \
  --max_encoder_length=3072 \
  --max_predictions_per_seq=75 \
  --masked_lm_prob=0.15 \
  --do_train=true \
  --do_eval=false \
  --do_export=false \
  --train_batch_size=4 \
  --batch_size=4 \
  --preprocessed_data=true