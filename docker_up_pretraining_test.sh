#!/bin/bash

cd /home/bigtree/hyungrak/bigbird/

/usr/local/bin/docker-compose run -d backend /app/ko_bigbird/bigbird/pretrain/run_pretraining_test.sh
