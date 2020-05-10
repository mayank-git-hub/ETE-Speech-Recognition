# End To End Speech Recognition model in PyTorch

This repository implements [Hybrid CTC/Attention Architecture for End-to-End Speech Recognition](https://ieeexplore.ieee.org/document/8068205)
in PyTorch. The architecture has been used as it is from [ESPNET](https://github.com/espnet/espnet), and the pre-processing steps have been modified and converted from
CPP to python.

# Cloning the repository

    $ git clone https://github.com/mayank-git-hub/ETE-Speech-Recognition
    $ cd ETE-Speech-Recognition
    $ pip install -r requirements.txt

# Downloading the data set

In specificConfig.py, set the path to where you would want to download the data set in "path_to_download".
axel should be installed for downloading the dataset. If you do not want to download it using axel, then download the tar.gz files and place it in the folder - 
config.path_to_download + '/' + list_i + '.tar.gz'

    $ python downloadDataset.py
    
# Generating the unigram

In specificConfig.py, set the path to where you would want to save the unigram model in "cache_dir".

    $ python main.py genUnigram
    
# Training the Automatic Speech Recognition Model

In specificConfig.py, set the path to where you would want to save the intermediate models in "base_model_path".

    $ python main.py train
    
# Continuing training the Automatic Speech Recognition Model from an intermediate model

In specificConfig.py, set the path to where you have the intermediate models in "resume\[model_path\]" and set "resume\[restart\]" to True.

    $ python main.py train
    
# Testing the Automatic Speech Recognition Model

In specificConfig.py, set the path to where you have the test model in "test_model".

    $ python main.py test