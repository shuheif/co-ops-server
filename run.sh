#!/bin/bash

# Set to your own path
# export CKPT_DIR=/Users/shuhei/Desktop/co-ops-server/checkpoints

docker build -t co-ops-server .
docker run -p 8080:8080 co-ops-server
