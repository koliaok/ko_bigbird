#!/bin/bash

cd /var/lib/docker/volumes/vol/_data/ko_bigbird/

/usr/local/bin/docker-compose run -d backend /app/ko_bigbird/bigbird/pretrain/run_pretraining_test.sh