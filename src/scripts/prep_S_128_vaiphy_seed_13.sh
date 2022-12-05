#!/bin/bash -l
#SBATCH -A snic2021-5-259
#SBATCH -n 4
#SBATCH -t 1:00:00
#SBATCH --reservation devel
#SBATCH -J prep_DS1
#SBATCH -e /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results/DS1/prep_S_128_vaiphy_seed_13_err.txt
#SBATCH -o /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results/DS1/prep_S_128_vaiphy_seed_13_out.txt
echo "$(date) Running on: $(hostname)"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/vbpi-nf/conda_vbpi_nf/
wait
echo "$(date) Modules / environments are loaded"

cd /proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/src
wait
echo "$(date) Directory is changed"

python post_analysis_prep.py --dataset DS1 --vaiphy_seed 13 --n_particles 128 --ng_stepsize 0.8 --max_iter 200 --init_strategy nj_phyml --samp_strategy slantis --branch_strategy ml 
wait
echo "$(date) Vaiphy is finished"

echo "$(date) All done!" >&2
