#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=rnn.kp20k.multi_test.general
#SBATCH --output=slurm_output/train.rnn.kp20k.multi_test.general.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules

# Run the job
export ATTENTION="general"
export EXP_NAME="rnn.general.plain_rnn_ref_stackof"
export ROOT_PATH="C:\Users\Tianyi Ma\Desktop\seq2seq-keyphrase-pytorch-master"
export DATA_NAME="stackof"
python -m train -data_path_prefix "data/$DATA_NAME/$DATA_NAME" -vocab_path "data/$DATA_NAME/$DATA_NAME.vocab.pt" -exp "$DATA_NAME" -exp_path "$ROOT_PATH/exp/$EXP_NAME/%s.%s" -batch_size 32 -bidirectional -run_valid_every 1000 -save_model_every 1000 -bidirectional -reinforce -attention_mode "$ATTENTION" -beam_size 16 -beam_search_batch_size 8 -train_ml -epoch 20
$SHELL