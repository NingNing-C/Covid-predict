# Covid-predict

## Components
- `data` :  Required data for training 
- `scripts` : Bash scripts
- `src` : python code 
- `analysis` : jupyter notebooks for all the analysis and plots


## System requirements
### Hardware requirements
Trained and tested on one NVIDIA Tesla V100 with 32GB GPU memory  

For storing all intermediate files for all methods and all datasets, approximately 100G of disk space will be needed.

### Software requirements

The codes have been tested on CentOS Linux release 7.9.2009 with conda 4.13.0 and python 3.8.5. The list of software dependencies are provided in the `environment.yml` file.
 
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
    conda install huggingface_hub=0.2.1 --force
```


## Model Download

The model could be download through the link https://drive.google.com/file/d/1IMeVKB41kakB3R5z_-xJPgfHCL7Axb72/view?usp=sharing

The model could be put under the folder trained_model 

## Usage
To train the model in 5-folds cross-validation, change $fold to 0-4 :
```
    bash scripts/train_model.sh $fold
```

To sythesize the high-risk variants, change $task_id to a number:
```
    bash scripts/synthetic.sh $task_id
```

