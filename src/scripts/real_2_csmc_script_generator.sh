#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/src"
RES_DIR="/proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results"

INIT_STRATEGY="nj_phyml"
SAMP_STRATEGY="slantis"
BRANCH_STRATEGY="ml"
NG_STEPSIZE=0.8

CSMC_N_REP=10
CSMC_BRANCH_STRATEGY='naive_w_labels'
CSMC_TREE_STRATEGY='bootstrap_2'

for EXP_ID in 1 2 3 4 5
do
	for NUM_PARTICLES in 128 
	do
		for VAIPHY_SEED in 13
		do
			for CSMC_NUM_PARTICLES in 4096
			do
				for POISSON_RATE in 0 4     
				do
					RESULT_DIR=${RES_DIR}
					EXP_NAME="DS${EXP_ID}"
					CSMC_NAME="S_${NUM_PARTICLES}_vaiphy_seed_${VAIPHY_SEED}_K_${CSMC_NUM_PARTICLES}_poisson_${POISSON_RATE}_rep_${CSMC_N_REP}_${CSMC_BRANCH_STRATEGY}_${CSMC_TREE_STRATEGY}"
					mkdir ${RESULT_DIR}/
					mkdir ${RESULT_DIR}/${EXP_NAME}

					job_file="${RESULT_DIR}/${EXP_NAME}/csmc_${CSMC_NAME}.sh"

					echo "#!/bin/bash -l
#SBATCH -A snic2021-5-259
#SBATCH -n 10
#SBATCH -t 12:00:00
#SBATCH -J csmc_${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/csmc_${CSMC_NAME}_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/csmc_${CSMC_NAME}_out.txt
echo \"\$(date) Running on: \$(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/vbpi-nf/conda_vbpi_nf/
wait
echo \"\$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR
wait
echo \"\$(date) Directory is changed\"

python post_sampling/main.py --dataset ${EXP_NAME} --vaiphy_seed ${VAIPHY_SEED} --n_particles ${NUM_PARTICLES} --ng_stepsize ${NG_STEPSIZE} --init_strategy ${INIT_STRATEGY} --samp_strategy ${SAMP_STRATEGY} --branch_strategy ${BRANCH_STRATEGY} --csmc_result_path ${RESULT_DIR}/ --csmc_n_particles ${CSMC_NUM_PARTICLES} --csmc_distortion_poisson_rate ${POISSON_RATE} --csmc_n_repetitions ${CSMC_N_REP} --csmc_branch_strategy ${CSMC_BRANCH_STRATEGY} --csmc_tree_strategy ${CSMC_TREE_STRATEGY} 
wait
echo \"\$(date) Vaiphy is finished\"

echo \"\$(date) All done!\" >&2" > $job_file

					sbatch $job_file
				done
			done
		done
	done
done
