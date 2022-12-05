#!/bin/bash -l

CODE_SRC_DIR="/proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/src"
RES_DIR="/proj/sc_ml/users/x_hazko/new_vaiphy/vaiphy/results"

INIT_STRATEGY="nj_phyml"
SAMP_STRATEGY="slantis"
BRANCH_STRATEGY="ml"
MAX_ITER=200
NG_STEPSIZE=0.8

for EXP_ID in 1 2 3 4 5
do
	for NUM_PARTICLES in 32 128
	do
		for VAIPHY_SEED in 13 42   
		do
			RESULT_DIR=${RES_DIR}
			EXP_NAME="DS${EXP_ID}"
			VAIPHY_NAME="S_${NUM_PARTICLES}_vaiphy_seed_${VAIPHY_SEED}"
			mkdir ${RESULT_DIR}/
			mkdir ${RESULT_DIR}/${EXP_NAME}

			job_file="${RESULT_DIR}/${EXP_NAME}/vaiphy_${VAIPHY_NAME}.sh"

			echo "#!/bin/bash -l
#SBATCH -A snic2021-5-259
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH -J vaiphy_${EXP_NAME}
#SBATCH -e ${RESULT_DIR}/${EXP_NAME}/vaiphy_${VAIPHY_NAME}_err.txt
#SBATCH -o ${RESULT_DIR}/${EXP_NAME}/vaiphy_${VAIPHY_NAME}_out.txt
echo \"\$(date) Running on: \$(hostname)\"

module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
source activate /proj/sc_ml/users/x_hazko/vbpi-nf/conda_vbpi_nf/
wait
echo \"\$(date) Modules / environments are loaded\"

cd $CODE_SRC_DIR
wait
echo \"\$(date) Directory is changed\"

python main.py --dataset ${EXP_NAME} --vaiphy_seed ${VAIPHY_SEED} --n_particles ${NUM_PARTICLES} --ng_stepsize ${NG_STEPSIZE} --max_iter ${MAX_ITER} --init_strategy ${INIT_STRATEGY} --samp_strategy ${SAMP_STRATEGY} --branch_strategy ${BRANCH_STRATEGY} 
wait
echo \"\$(date) Vaiphy is finished\"

echo \"\$(date) All done!\" >&2" > $job_file

			sbatch $job_file
		done
	done
done
