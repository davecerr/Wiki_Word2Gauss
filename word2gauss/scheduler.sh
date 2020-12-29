# This is an example of a job script.
#$ -S /bin/bash
#$ -l h_rt=0:10:0
#$ -l h_vmem=2G,tmem=2G
#$ -cwd
#$ -j y
#$ -N Test W2G Embedding


# Commands to be executed go here...


python main.py --MWE=2 --dynamic_window_size=True --report_schedule=1000 --num_epochs=2 --dim=5 --neg_samples=2 --eta=0.1 --Closs=0.1 --save=False --csv=True
