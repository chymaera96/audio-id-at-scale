#!/usr/bin/env bash

python benchmark.py --fp_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --dummy_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --test_ids=500 --num_dummy=10000 --iterations=20 --mean=0.000703 --std=0.088386
python benchmark.py --fp_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --dummy_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --test_ids=500 --num_dummy=50000 --iterations=20 --mean=0.000703 --std=0.088386
python benchmark.py --fp_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --dummy_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --test_ids=500 --num_dummy=100000 --iterations=20 --mean=0.000703 --std=0.088386
python benchmark.py --fp_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --dummy_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --test_ids=500 --num_dummy=500000 --iterations=20 --mean=0.000703 --std=0.088386 
python benchmark.py --fp_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --dummy_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --test_ids=500 --num_dummy=1000000 --iterations=20 --mean=0.000703 --std=0.088386 
python benchmark.py --fp_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --dummy_dir=/data/scratch/acw723/logs/emb/large/model_nmfp --test_ids=500 --num_dummy=6000000 --iterations=20 --mean=0.000703 --std=0.088386 