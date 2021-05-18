# Sign Language Tranalation (Multi-tasking transformer)

This repo contains the training and evaluation code for the paper [End-To-End Sign Language Translation via
Multitask Learning]. 

This code is based on [SLT](https://github.com/neccam/slt) but modified to realize our objective. 
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [ ] (Nice to have) - Proper guide to run the code .




