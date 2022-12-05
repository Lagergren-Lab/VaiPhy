#!/bin/bash -l
#SBATCH -A snic2021-5-258
#SBATCH -n 10
#SBATCH -t 24:00:00
#SBATCH -J csmc_DS1
#SBATCH -e /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results/DS1/csmc_S_128_vaiphy_seed_13_K_2048_poisson_0_rep_10_same_parent_naive_bootstrap_0_err.txt
#SBATCH -o /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results/DS1/csmc_S_128_vaiphy_seed_13_K_2048_poisson_0_rep_10_same_parent_naive_bootstrap_0_out.txt
echo "$(date) Running on: $(hostname)"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/vbpi-nf/conda_vbpi_nf/
wait
echo "$(date) Modules / environments are loaded"

cd /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/src
wait
echo "$(date) Directory is changed"

python post_sampling/main.py --dataset DS1 --vaiphy_seed 13 --n_particles 128 --ng_stepsize 0.8 --init_strategy nj_phyml --samp_strategy slantis --branch_strategy ml --csmc_result_path /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results/ --csmc_n_particles 2048 --csmc_distortion_poisson_rate 0 --csmc_n_repetitions 10 --csmc_branch_strategy same_parent --csmc_tree_strategy naive_bootstrap_0 
wait
echo "$(date) Vaiphy is finished"

echo "$(date) All done!" >&2
