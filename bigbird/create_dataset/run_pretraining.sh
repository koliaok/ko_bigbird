#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 bigbird/create_dataset/create_pretraining_data.py \
  --input_file=./training_data.txt \
  --output_file=../dataset \
  --vocab_file=../vocab/gpt2.model \
  --max_seq_length=3072 \
  --max_predictions_per_seq=75 \
  --random_seed=12345 \
  --masked_lm_prob=0.15 \
