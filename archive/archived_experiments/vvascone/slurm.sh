#!/bin/bash
#Set job requirements (note set time 1.5 to 2x longer than expected)
#SBATCH -t 96:00:00
#SBATCH -N 1
#SBATCH --ntasks=16

#Send email at start en end
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bas.chatel@radboudumc.nl

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0-bare
# module load SciPy-bundle/2022.05-foss-2022a

python $HOME/run_sims.py