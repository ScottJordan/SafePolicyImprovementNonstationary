# SafePolicyImprovementNonstationary
This a repository containing the code for the paper "Towards Safe Policy Improvement forNon-Stationary MDPs"

For dependencies, see the file `Project.toml`.

### Setup Conda environment for Julia
```julia

ENV["PYTHON"] = "/path/to/miniconda3/envs/<env_name>/bin/python"
] build PyCall
build IJulia

add https://github.com/ScottJordan/EvaluationOfRLAlgs.git 
```



### Install Glucose Simulator


```bash
conda activate <env_name> #Activate the virtual specified above. 
cd python/SimGlucose
pip install -e .
```

If you get MKL errors when trying to use the simulator from Julia. Uninstall the conda numpy library that has MKL (should be the default) and then add the one without MKL. 

To reproduce results in the paper run the files `experiments/bandit_swarm.jl` and `experiments/glucose_swarm.jl`. 

The jupyter notebook `experiments/plots.ipynb` contains code for plotting and analyzing the results. 