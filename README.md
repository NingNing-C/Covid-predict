# Covid-predict

## Components
- `data` :  Required data
- `scripts` : Bash scripts
- `src` : python code 
- `analysis` : jupyter notebooks for all the analysis and plots
- `result` : directory for the results


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
## Data Download
GISAD dataset repuires authentication, and registration is needed to access the data. Therefore, we can't provide the data directly. You can download the data from their web: https://www.gisaid.org. 

## Model Download

The model could be download through the link https://drive.google.com/file/d/1IMeVKB41kakB3R5z_-xJPgfHCL7Axb72/view?usp=sharing

The model could be put under the folder trained_model 

## Usage
Here, we provide an example of model inference and variants synthetic using the circulating variants `data/pVNT_seq.csv`. You will find the results in `result` directory after running the below commands.

To train the model in 5-folds cross-validation, change $fold to 0-4 :
```
    bash scripts/train_model.sh $fold
```
To get the predictions and embeddings for variants :

```
    bash scripts/run_inference.sh
```


To sythesize the high-risk variants:
```
    bash scripts/synthetic.sh
```

