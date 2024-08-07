#!/bin/bash
#SBATCH --mem=64g                              # required number of memory
#SBATCH --nodes=1                               # nodes required for whole simulation <-- determine from architecture and distribution of GPUs/CPUs

#SBATCH --ntasks-per-node=1                    # number of tasks/clients per node
#SBATCH --cpus-per-task=32                       # CPUs for each task/client
#SBATCH --gpus-per-task=1                       # GPUs for each task/client

#SBATCH --partition=gpuA100x4                    # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

#SBATCH --gpus-per-node=1                       # number of GPUs you want to use on 1 node
##SBATCH --gpu-bind=none                         # in APACE we used closest

#SBATCH --account=bbjo-delta-gpu               # <- one of: bbke-delta-cpu or bbke-delta-gpu

#SBATCH --job-name=MNIST_grpc_client1A100_FedCompass    # job name
#SBATCH --time=00:30:00                         # dd-hh:mm:ss for the job

#SBATCH -e MNIST_grpc_client1A100__FedCompass-err-%j.log
#SBATCH -o MNIST_grpc_client1A100__FedCompass-out-%j.log

#SBATCH --constraint="scratch"

#SBATCH --mail-user=pp32@illinois.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options



source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda deactivate
conda deactivate  # just making sure
module purge
module reset  # load the default Delta modules

module load anaconda3_gpu
module list

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda activate /u/parthpatel7173/.conda/envs/APPFL


cd /scratch/bcbw/parthpatel7173/FedCompass/examples


# Start the client -- Before starting the client, set server IP and port in config

python3 ./grpc/run_client_1.py
