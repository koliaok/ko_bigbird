#!/bin/bash

cd /home/bigtree/hyungrak/bigbird/

/usr/local/bin/docker-compose run backend /app/ko_bigbird/bigbird/pretrain/run_pretraining_create_serve_model.sh
