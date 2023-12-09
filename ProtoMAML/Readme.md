Download the dataset:
```
wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
```
Unzip the files:
```
unzip -qq images_background.zip
unzip -qq images_evaluation.zip
```

# Reproducing training runs

The slurm job files corresponding to the training runs are in the jobs directory.

Create a conda environment sml_project (used in slurm job) and install the requirements.txt file.
```
conda create --name sml_project python=3.6

conda activate sml_project

pip install -r requirements.txt
```
To run the slurm job file, use the following command:
```
sbatch job_20_1
sbatch job_5_1
sbatch job_20_1
sbatch job_5_5
```
