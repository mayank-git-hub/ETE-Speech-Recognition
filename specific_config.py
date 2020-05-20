import os

path_to_download = ''
base_model_path = ''

os.makedirs(path_to_download, exist_ok=True)
os.makedirs(base_model_path, exist_ok=True)

test_model = base_model_path + ''
cache_dir = ''

resume = {
	'restart': False,
	'model_path':
		''
}

num_cuda = '0'

os.environ["CUDA_VISIBLE_DEVICES"] = num_cuda
