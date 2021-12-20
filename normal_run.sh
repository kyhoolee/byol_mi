#!/bin/bash
python3 main.py --nodes 1 --gpus 8 --loss_type normal --train_dir ./train_dir_20211220_normal
python3 logistic_regression.py --model_path train_dir_20211220_normal/model-final.pt