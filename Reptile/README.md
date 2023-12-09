# Getting the data

The [fetch_data.sh](fetch_data.sh) script creates a `data/` directory and downloads Omniglot into it. 

```
$ ./fetch_data.sh
Fetching omniglot/images_background ...
Extracting omniglot/images_background ...
Fetching omniglot/images_evaluation ...
Extracting omniglot/images_evaluation ...
...
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
sbatch jobs/job_1_shot_5_way
sbatch jobs/job_5_shot_5_way
sbatch jobs/job_1_shot_20_way
sbatch jobs/job_5_shot_20_way
```

<!-- references -->
[1]: On First-Order Meta-Learning Algorithms  https://arxiv.org/abs/1803.02999
@misc{nichol2018firstorder,
      title={On First-Order Meta-Learning Algorithms}, 
      author={Alex Nichol and Joshua Achiam and John Schulman},
      year={2018},
      eprint={1803.02999},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
