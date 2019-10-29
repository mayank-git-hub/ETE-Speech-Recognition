import os
from datetime import datetime
from specific_config import *
import shutil

list_to_download = [
	'dev-clean', 'test-clean', 'dev-other', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
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
	'nfilt': 80,
	'rate': 16000,
}

rmlsutt = {
	'max_frames': 3000,
	'min_frames': 10,
	'max_chars': 400,
	'min_chars': 0,
}

seed = 1
num_epochs = 10
CTC_LOSS_THRESHOLD = 10000

model_save_path = base_model_path + str(datetime.time(datetime.now()))
os.makedirs(model_save_path, exist_ok=True)
shutil.copyfile('config.py', model_save_path + '/config.py')

use_cuda = True

train_param = {
	'batch_size': 10,
	'num_workers': 8,
	'lr': 5.0,
	'adim': 256,
	'transformer_warmup_steps': 25000,
}

test_param = {
	'batch_size': 10,
	'num_workers': 8,
}


class ModelArgs:
	accum_grad = 8
	aconv_chans = -1
	aconv_filts = 100
	adim = 256
	aheads = 4
	apply_uttmvn = True
	atype = 'dot'
	awin = 5
	backend = 'pytorch'
	badim = 320
	batch_bins = 5992000
	batch_count = 'bin'
	batch_frames_in = 0
	batch_frames_inout = 0
	batch_frames_out = 0
	batch_size = 0
	bdropout_rate = 0.0
	beam_size = 4
	blayers = 2
	bprojs = 300
	btype = 'blstmp'
	bunits = 300
	config = 'conf/train.yaml'
	config2 = None
	config3 = None
	context_residual = False
	criterion = 'acc'
	ctc_type = 'warpctc'
	ctc_weight = 0.3
	debugdir = 'exp/train_960_pytorch_train_specaug'
	debugmode = 1
	dict = 'data/lang_char/train_960_unigram5000_units.txt'
	dlayers = 6
	dropout_rate = 0.1
	dropout_rate_decoder = 0.0
	dtype = 'lstm'
	dunits = 2048
	early_stop_criterion = 'validation/main/acc'
	elayers = 12
	elayers_sd = 4
	epochs = 100
	eprojs = 320
	eps = 1e-08
	eps_decay = 0.01
	etype = 'blstmp'
	eunits = 2048
	fbank_fmax = None
	fbank_fmin = 0.0
	fbank_fs = 16000
	grad_clip = 5.0
	grad_noise = False
	lm_weight = 0.1
	lsm_type = ''
	lsm_weight = 0.1
	maxlen_in = 800
	maxlen_out = 150
	maxlenratio = 0.0
	minibatches = 0
	minlenratio = 0.0
	model_module = 'espnet.nets.pytorch_backend.e2e_asr_transformer:E2E'
	mtlalpha = 0.3
	n_iter_processes = 0
	n_mels = 80
	nbest = 1
	ngpu = 1
	num_save_attention = 3
	num_spkrs = 1
	opt = 'noam'
	outdir = 'exp/train_960_pytorch_train_specaug/results'
	patience = 0
	penalty = 0.0
	preprocess_conf = 'conf/specaug.yaml'
	ref_channel = -1
	report_cer = False
	report_wer = False
	resume = None
	rnnlm = None
	rnnlm_conf = None
	sampling_probability = 0.0
	seed = 1
	sortagrad = 0
	spa = False
	stats_file = None
	subsample = '1'
	sym_blank = '<blank>'
	sym_space = '<space>'
	tensorboard_dir = 'tensorboard/train_960_pytorch_train_specaug'
	threshold = 0.0001
	train_json = 'dump/train_960/deltafalse/data_unigram5000.json'
	transformer_attn_dropout_rate = 0.0
	transformer_init = 'pytorch'
	transformer_input_layer = 'conv2d'
	transformer_length_normalized_loss = 0
	transformer_lr = 5.0
	transformer_warmup_steps = 25000
	use_beamformer = True
	use_dnn_mask_for_wpe = False
	use_frontend = False
	use_wpe = False
	uttmvn_norm_means = True
	uttmvn_norm_vars = False
	valid_json = 'dump/dev/deltafalse/data_unigram5000.json'
	verbose = 0
	wdropout_rate = 0.0
	weight_decay = 0.0
	wlayers = 2
	wpe_delay = 3
	wpe_taps = 5
	wprojs = 300
	wtype = 'blstmp'
	wunits = 300
