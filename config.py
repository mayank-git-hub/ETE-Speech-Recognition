list_to_download = [
		'dev-clean', 'test-clean', 'dev-other', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
path_to_download = '/media/mayank/0b40607e-7efc-4216-b12f-8bb86facfaed/Dataset_HDD/Audio/LibriSpeech/LibriSpeech'
cache_dir = 'Cache'
url_base = 'www.openslr.org/resources/12/'

train_set = ['train-clean-100', 'train-clean-360', 'train-other-500']
train_dev = ['dev-clean', 'dev-other']
recog_set = ['test-clean', 'test-other', 'dev-clean', 'dev-other']

fbank = {
	'pre_emphasis': 0.97,
	'frame_size': 0.025,
	'frame_stride': 0.01,
	'n_fft': 512,
	'nfilt': 40,
	'rate': 16000,
}

rmlsutt = {
	'max_frames': 3000,
	'min_frames': 10,
	'max_chars': 400,
	'min_chars': 0,
}