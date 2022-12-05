#!/bin/bash -l
#SBATCH -A snic2021-5-259
#SBATCH -n 4
#SBATCH -t 03:00:00
#SBATCH -J csmc_sim1
#SBATCH -e /proj/sc_ml/users/x_harme/vaiphy/results/sim1/err.txt
#SBATCH -o /proj/sc_ml/users/x_harme/vaiphy/results/sim1/out.txt
echo "$(date) Running on: $(hostname)"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/vbpi-nf/conda_vbpi_nf/
wait
echo "$(date) Modules / environments are loaded"

cd /home/x_harme/vaiphy/src
wait
echo "$(date) Directory is changed"

export CODE_SRC_DIR="/home/x_harme/vcsmc"
export NUM_PARTICLES=30
export NUM_PARTICLES=30
export DATASET = "DS1"
export MAX_ITER = 50
export SAMP_STRATEGY = "csmc"
export SLANTIS_EXPLORE_RATE = 0.5

# CSMC stuff
export CSMC_RES_DIR = "/proj/sc_ml/users/x_harme/vaiphy/results/"
export CSMC_N_PARTICLES = 256
export CSMC_PO_DIST_RATE = 0
export CSMC_N_SEEDS = 3
export CSMC_BRANCH_STRAT = "naive_w_labels"
export CSMC_TREE_STRAT = "bootstrap_0"

python main.py --dataset DATASET --vaiphy_seed 13 --n_particles NUM_PARTICLES --ng_stepsize 0.8 --max_iter MAX_ITER --init_strategy nj_phyml --samp_strategy SAMP_STRATEGY --branch_strategy ml --slantis_explore_rate SLANTIS_EXPLORE_RATE
wait
echo "$(date) Vaiphy is finished"

echo "$(date) All done!" >&2