#!/bin/bash
python3 main.py --nodes 1 --gpus 4 --loss_type mi --train_dir ./train_dir_20211220_mi
python3 logistic_regression.py --model_path train_dir_20211220_mi/model-final.pt