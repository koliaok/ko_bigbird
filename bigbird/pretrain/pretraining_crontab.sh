00 23 * * * /app/ko_bigbird/bigbird/pretrain/down.sh >> /app/ko_bigbird/bigbird/pretrain/log_down.txt 2>&1
30 10 * * * /app/ko_bigbird/bigbird/pretrain/run_pretraining.sh >> /app/ko_bigbird/bigbird/pretrain/log_excution.txt 2>&1