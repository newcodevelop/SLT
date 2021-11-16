# Sign Language Tranalation 



This code is based on [SLT](https://github.com/neccam/slt) but modified to realize our objective. Please note that this code is highly messy and we aim to make it cleaned up and maintainable once our paper is accepted. [Some Features May Not Work.]
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
! In the model file (`./signjoey/model.py`) the logic for our proposed model is present. To change loss weight, we have to manually change them in the file itself.
! The decoders file (`./signjoey/decoders.py`) contain code for our proposed model's decoding section.


