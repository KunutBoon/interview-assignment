# Home Credit Risk challenge
This challenge from Home Credit invited all Kagglers to unlock the full potential of data by leveraging multiple datasets to predict the probability of payment difficulties within a determined period. As part of the interview assignment to create Data Science workflow on this problem, here are some introduction of my source code related to this workflow.

## Project File Structures
The structure of files in this repository are illustrated below
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

First of all, in order to execute successfully, the python environment to run the code is required with library dependencies listed in <code>requirements.txt</code> files. Please make sure that you set up the environment correctly with pre-installed all libraries before running any code (<a href="https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python">Conda environment</a> is preferred)

## Data File and Folder Structures
Next, all dataset from the Kaggle challenge are needed as a input of this workflow. Please note that all raw datasets must be stored inside the given directory
```
ROOT
├── data
│   └── raw                                   <- All the data must put in this directory
│         └── application_test.csv
│         └── application_train.csv
│         └── ....
├── config/training
├── notebook
├── src
├── .gitignore
└── ....
```

The name of data used in the source code are the same as its original names. Therefore, no modification is needed.

Also, it worths to mentioned that there might be some process to create another directories from the source code itself. So, please make sure that the program or the directory have the right to create/read/write the directory and files

## File Submission
There are 2 files related to this submission, which are
 - credit_defult_risk_experiment.ipynb : This Jupyter notebook contains all the logics and processes which have been used during the analysis and ML model experimental phase
 - model_prediction_workflow.ipynb : This Jupyter notebook contains all the code related to all pipelines, including Data Prep and Model Training Pipeline and prediction pipeline. It basically call 2 other python files at the root directory.

All related files could be listed as below
 ```
ROOT
├── notebook
│   ├── credit_defult_risk_experiment.ipynb         <- Experimental Phase
│   └── model_prediction_workflow.ipynb             <- Pipeline execution
├── ...
├── pipeline_train.py ** called by notebook **
├── pipeline_inference.py ** called by notebook **
└── ...
```

If there is any question or any issues, please do not hesitate to contact me via email. Thank You !!!