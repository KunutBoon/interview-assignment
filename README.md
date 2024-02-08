# Home Credit Risk challenge
This challenge from Home Credit invited all Kagglers to unlock the full potential of data by leveraging multiple datasets to predict the clent's payment difficulties within a specified period. As part of the assignment to create Data Science workflow on this challenge, here are some introduction of my source code involved with this workflow.

## Project File Structures
The structure of files in this repository is illustrated below
```
ROOT
├── config/training                           <- Folder for model training config
│   └── hyper_param_config.json                     <- Config file for model training, noting but model hyparameters and learning setting
├── notebook                                  <- Folder for jupyter notebooks
│   ├── credit_defult_risk_experiment.ipynb         <- Jupyter notebooks related to model experiments
│   └── model_prediction_workflow.ipynb             <- Jupyter notebooks related to ML Pipeline
├── src                                       <- Folder for well-organized source code
│   ├── data_preprocessing.py                       <- .py file for pre-processing code
│   ├── model_evaluation.py                         <- .py file for model eval code
│   ├── model_training.py                           <- .py file for model training code
│   └── utils.py                                    <- .py file for utilities functions
├── .gitignore
├── README.md
├── pipeline_train.py                         <- .py file for model training pipeline
├── pipeline_inference.py                     <- .py file for model inference pipeline
└── requirements.txt                          <- requirement files for library dependencies
```
## System Settings

First of all, the python environment is required with libraries listed inside the <code>requirements.txt</code> files in order to successfully execute all the codes. Please make sure that the environment was set up correctly before any execution of the code.

Here are the versions of Conda environment which was used to complete the assignment
```
conda == 23.5.2
python == 3.9.18
```

To easily manage all dependencies, Conda virtual environment is recommended. You can use this command to manage the environment after installing conda (Miniconda 3)

```
conda create --name <env_name> --file requirements.txt    ## create the environment with requirement.txt
conda list                                                ## list all imported packages
conda activate <env_name>                                 ## activate the environment
conda deactivate                                          ## deactivate the environment
```

## Input Data File Structures
Next, all given dataset from the Kaggle challenge are needed as a input of this workflow. All raw datasets must be stored inside the data/raw folder since the source code will try to access data from that particular folder. Please refer to full-detail information below
```
ROOT
├── data
│   └── raw  ** All the data must be stored inside this directory **
│         └── <file_1>.csv
│         └── <file_2>.csv
│         └── ....
├── config/training
├── notebook
├── src
├── .gitignore
└── ....
```

The name of dataset used in the source code are the same as original. No modification on file name is needed.

Also, it worths to mentioned that there might be some additional directories created by the source code itself. Please make sure that the program or the directory itself have the sufficient permission to create/read/write some directories and files

## Submission Files
There are 2 Jupyter Notebook files summitted in this repository, including
 - <code>credit_defult_risk_experiment.ipynb</code> : The Jupyter notebook file contains all the code involved with all of Data Analysis tasks and processes in the Data Science workflow
 - <code>model_prediction_workflow.ipynb</code> : The Jupyter notebook contains commands related to executions of all model prediction workflows, including Model Training pipeline and Model Prediction pipeline. Generally, each command calls 2 other .py files for pipeline execution.

All related files could be listed below
 ```
ROOT
├── notebook
│   ├── credit_defult_risk_experiment.ipynb         <- .ipynb file involved with Data Science experiments
│   └── model_prediction_workflow.ipynb             <- .ipynb file involved with model prediction workflows
├── ...
├── pipeline_train.py ** called by 2nd notebook **
├── pipeline_inference.py ** called by 2nd notebook **
└── ...
```

As the order of code execution in each Jupyter notebook is designed in sequence, the execution of code could be done cell by cell from the beginning to inspec the output.

If there is any question or issues, please do not hesitate to contact me via email. I would be happy to answer those quesions.