path_to_download = '/home/SharedData/Mayank/Audio/LibriSpeech'
base_model_path = '/home/SharedData/Mayank/Audio/models/'

test_model = base_model_path + '/16:24:30.041201/1'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"