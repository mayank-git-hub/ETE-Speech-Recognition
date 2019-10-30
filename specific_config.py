path_to_download = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/Dataset_HDD/Audio/LibriSpeech/LibriSpeech'
base_model_path = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/Dataset_HDD/Audio/LibriSpeech/models/'
test_model = base_model_path + '10:37:25.068170/LibriSpeech_train960.0.1130.8476.10:37:40.972593.pth'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"