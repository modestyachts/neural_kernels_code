#!/bin/bash
ulimit -n 99999
pushd ..
python run_train_eval_exp.py --config configs/myrtle5.yaml   --config-updates DATASET.TRAIN_SUBSET 50000 DATASET.TEST_SUBSET 10000 DATASET.NAME  "cifar-10"  SYSTEM.BATCH_SIZE 48
