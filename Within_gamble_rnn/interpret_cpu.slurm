#!/bin/bash
#SBATCH --job-name=rnn_interprete
#SBATCH --account=ds_ga_1006-2025fa
#SBATCH --partition=interactive
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay  /scratch/zf2345/overlay-25GB-500K.ext3:rw /scratch/zf2345/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate rnn

cd /scratch/zf2345/hurd_rnn/Within_gamble_rnn/
python interprete_main.py \
    --model_path /scratch/zf2345/results/20250121_143022/rnn_model.pt \
    --obj_path /scratch/zf2345/hurd_rnn/Data/c13k_obj_feats_uid.csv \
    --subj_path /scratch/zf2345/hurd_rnn/Data/c13k_subject_data_uid.csv \
    --output_dir /scratch/zf2345/hurd_rnn/Within_gamble_rnn/interpretation_results \
    --batch_size 256 \
    --device cpu
"