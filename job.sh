#!/usr/bin/env bash


nohup python ./dft_for_ising/train.py  --hidden_channel 40 40 40 40 40 40 40 40  --kernel_size 3 3 --padding 1 1  --model_name=model_test_10k --data_path=data/datasets/dataset_fullyconnected_spinglass_ndata_10000_nsites_16_sigma_0.5.npz --model_type=REDENTnopooling2D --padding_mode='zeros' --pooling_size=1 --epochs=3000 > output_train_2.txt &

