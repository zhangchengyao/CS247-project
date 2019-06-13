#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=predict_dag.beamsize
#SBATCH --output=predict_dag.out.beamsize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
#module restore


# Run the job

#SBATCH --output=predict_dag${beam_size}.out
export EXP_NAME="plain_rnn_ref_pred"
export DATA_NAME="stackof"
export ATTENTION="general"
export MODEL_NAME="exp/rnn.general.plain_rnn_ref_stackof/stackof.ml.20190602-023636/model/stackof.ml.epoch=4.batch=5674.total_batch=25000.model"
python -m predict -data_path_prefix "data/$DATA_NAME/$DATA_NAME" -train_from "$MODEL_NAME" -vocab_path "data/$DATA_NAME/$DATA_NAME.vocab.pt" -exp_path "exp/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -bidirectional -batch_size 64 -beam_search_batch_size 32  -batch_workers 2 -beam_size 32 -train_ml -copy_mode "$ATTENTION" -attention_mode "$ATTENTION"
$SHELL
