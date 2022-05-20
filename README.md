# Covid-predict

## Components
- `data` :  Required data for training 
- `scripts` : Bash scripts
- `src` : python code 


## Installation

1. Create the conda environment from the environment.yaml file:
```
    conda env create -f environment.yml
```

2. Activate the new conda environment:
```
    conda activate covid_predict
```
3. Update huggingface_hub package
```
    conda install huggingface_hub=0.2.1
```

## Usage
To train the model in 5-folds cross-validation, change $fold to 0-4 :
```
    bash scripts/train-model.sh $fold
```

To sythesize the high-risk variants, change $task_id to a number:
```
    bash scripts/synthetic.sh $task_id
```
