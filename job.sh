#!/usr/bin/env bash


nohup python ./dft_for_ising/train.py  --hidden_channel 40 40 40 40   --kernel_size 5 5 --padding 2 2  --model_name=model_test_50k --data_path=data/datasets/dataset_for_training_ndata_50000_nsites_16_test0.npz --model_type=REDENTnopooling2D --pooling_size=1 --epochs=3000 > output_train.txt &

