#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2

export PYTHONPATH='/app/ko_bigbird/'

poetry run python ./create_pretraining_data.py \
  --input_file=../datasource/pretrained_data/pretraining_data.txt \
  --output_file=../datasource/tf_pretrained_data/pretraining_data.tf_record \
  --vocab_file=../bpe_model/ko_big_bird.model \
  --max_seq_length=3072 \
  --max_predictions_per_seq=75 \
  --random_seed=12345 \
  --masked_lm_prob=0.15 \
  --split_output_data_len=6 \
