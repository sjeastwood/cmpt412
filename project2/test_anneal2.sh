#!/bin/bash
python3 train_lr_gamma_6.py >> lr_Anneal_06.txt
python3 train_lr_gamma_8.py >> lr_Anneal_08.txt
python3 train_lr_gamma_9.py >> lr_Anneal_09.txt
python3 train_lr_gamma_5_step15.py >> lr_Anneal_step15_gamma5.txt