import torch
from torch import nn
import config
import math
import numpy as np
from argparse import Namespace
import editdistance
from distutils.util import strtobool
from itertools import groupby
import torch.nn.functional as F

from pytorch_backend.ctc import CTC
from pytorch_backend.nets_utils import make_pad_mask
from pytorch_backend.nets_utils import th_accuracy
from pytorch_backend.transformer.attention import MultiHeadedAttention
from pytorch_backend.transformer.decoder import Decoder
from pytorch_backend.transformer.encoder import Encoder
from pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from pytorch_backend.transformer.layer_norm import LayerNorm

low_freq_mel = 0
high_freq_mel = 2595 * math.log10(1 + (config.fbank['rate'] / 2) / 700)  # Convert Hz to Mel
mel_points = torch.linspace(low_freq_mel, high_freq_mel, config.fbank['nfilt'] + 2)  # Equally spaced in Mel scale
hz_points = 700 * (torch.pow(10, mel_points / 2595) - 1)  # Convert Mel to Hz

bin_ = torch.floor((config.fbank['n_fft'] + 1) * hz_points / config.fbank['rate']).float()
fbank = torch.zeros((config.fbank['nfilt'], int(math.floor(config.fbank['n_fft'] / 2 + 1)))).float()

if config.use_cuda:
	fbank = fbank.cuda()
	bin_ = bin_.cuda()

for m in range(1, config.fbank['nfilt'] + 1):

	f_m_minus = int(bin_[m - 1])  # left
	f_m = int(bin_[m])  # center
	f_m_plus = int(bin_[m + 1])  # right

	index_1 = torch.arange(f_m_minus, f_m)
	index_2 = torch.arange(f_m, f_m_plus)

	if config.use_cuda:
		index_1 = index_1.cuda()
		index_2 = index_2.cuda()

	fbank[m - 1, index_1] = ((index_1.float() - bin_[m - 1]) / (
			bin_[m] - bin_[m - 1])).float()
	fbank[m - 1, index_2] = ((bin_[m + 1] - index_2.float()) / (
			bin_[m + 1] - bin_[m])).float()


def subsequent_mask(size, device="cpu", dtype=torch.uint8):
	"""Create mask for subsequent steps (1, size, size)

	:param int size: size of mask
	:param str device: "cpu" or "cuda" or torch.Tensor.device
	:param torch.dtype dtype: result dtype
	:rtype: torch.Tensor
	>>> subsequent_mask(3)
	[
		[1, 0, 0],
		[1, 1, 0],
		[1, 1, 1]
	]
	"""

	ret = torch.ones(size, size, device=device, dtype=dtype)
	return torch.tril(ret, out=ret)


def end_detect(ended_hyps, i, m=3, d_end=np.log(1 * np.exp(-10))):
	"""End detection

	described in Eq. (50) of S. Watanabe et al
	"Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"

	:param ended_hyps:
	:param i:
	:param m:
	:param d_end:
	:return:
	"""
	if len(ended_hyps) == 0:
		return False
	count = 0
	best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
	for m in range(m):
		# get ended_hyps with their length is i - m
		hyp_length = i - m
		hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
		if len(hyps_same_length) > 0:
			best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
			if best_hyp_same_length['score'] - best_hyp['score'] < d_end:
				count += 1

	if count == m:
		return True
	else:
		return False


class CTCPrefixScore(object):
	"""Compute CTC label sequence scores

	which is based on Algorithm 2 in WATANABE et al.
	"HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
	but extended to efficiently compute the probablities of multiple labels
	simultaneously
	"""

	def __init__(self, x, blank, eos, xp):
		self.xp = xp
		self.logzero = -10000000000.0
		self.blank = blank
		self.eos = eos
		self.input_length = len(x)
		self.x = x

	def initial_state(self):
		"""Obtain an initial CTC state

		:return: CTC state
		"""
		# initial CTC state is made of a frame x 2 tensor that corresponds to
		# r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
		# superscripts n and b (non-blank and blank), respectively.
		r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
		r[0, 1] = self.x[0, self.blank]
		for i in range(1, self.input_length):
			r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
		return r

	def __call__(self, y, cs, r_prev):

		cs = cs.cpu()
		"""Compute CTC prefix scores for next labels

		:param y     : prefix label sequence
		:param cs    : array of next labels
		:param r_prev: previous CTC state
		:return ctc_scores, ctc_states
		"""
		# initialize CTC states
		output_length = len(y) - 1  # ignore sos
		# new CTC states are prepared as a frame x (n or b) x n_labels tensor
		# that corresponds to r_t^n(h) and r_t^b(h).
		r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
		xs = self.x[:, cs]
		if output_length == 0:
			r[0, 0] = xs[0]
			r[0, 1] = self.logzero
		else:
			r[output_length - 1] = self.logzero

		# prepare forward probabilities for the last label
		r_sum = self.xp.logaddexp(r_prev[:, 0], r_prev[:, 1])  # log(r_t^n(g) + r_t^b(g))
		last = y[-1]
		if output_length > 0 and last in cs:
			log_phi = self.xp.ndarray((self.input_length, len(cs)), dtype=np.float32)
			for i in range(len(cs)):
				log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
		else:
			log_phi = r_sum

		# compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
		# and log prefix probabilites log(psi)
		start = max(output_length, 1)
		log_psi = r[start - 1, 0]
		for t in range(start, self.input_length):
			r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
			r[t, 1] = self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
			log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

		# get P(...eos|X) that ends with the prefix itself
		eos_pos = self.xp.where(cs == self.eos)[0]
		if len(eos_pos) > 0:
			log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

		# return the log prefix probability and CTC states, where the label axis
		# of the CTC states is moved to the first axis to slice it easily
		return log_psi, self.xp.rollaxis(r, 2)


class ASRInterface(object):
	"""ASR Interface for ESPnet model implementation"""

	@staticmethod
	def add_arguments(parser):
		return parser

	def forward(self, audio, audio_length, ys):
		# xs, ilens are computed from audio using the pre_process model
		"""compute loss for training

		:param xs:
			For pytorch, batch of padded source sequences torch.Tensor (B, Tmax, idim)
			For chainer, list of source sequences chainer.Variable
		:param ilens: batch of lengths of source sequences (B)
			For pytorch, torch.Tensor
			For chainer, list of int
		:param ys:
			For pytorch, batch of padded source sequences torch.Tensor (B, Lmax)
			For chainer, list of source sequences chainer.Variable
		:return: loss value
		:rtype: torch.Tensor for pytorch, chainer.Variable for chainer
		"""
		raise NotImplementedError("forward method is not implemented")

	def recognize(self, x, recog_args, char_list=None, rnnlm=None):
		"""recognize x for evaluation

			:param ndarray x: input acouctic feature (B, T, D) or (T, D)
			:param namespace recog_args: argment namespace contraining options
			:param list char_list: list of characters
			:param torch.nn.Module rnnlm: language model module
			:return: N-best decoding results
			:rtype: list

		"""

		raise NotImplementedError("recognize method is not implemented")

	def calculate_all_attentions(self, xs, ilens, ys):
		"""attention calculation

			:param list xs_pad: list of padded input sequences [(T1, idim), (T2, idim), ...]
			:param ndarray ilens: batch of lengths of input sequences (B)
			:param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
			:return: attention weights (B, Lmax, Tmax)
			:rtype: float ndarray
		"""
		raise NotImplementedError("calculate_all_attentions method is not implemented")


class ErrorCalculator(object):
	"""Calculate CER and WER for E2E_ASR and CTC models during training

	:param y_hats: numpy array with predicted text
	:param y_pads: numpy array with true (target) text
	:param char_list:
	:param sym_space:
	:param sym_blank:
	:return:
	"""

	def __init__(self, char_list, sym_space, sym_blank, report_cer=False, report_wer=False):
		super(ErrorCalculator, self).__init__()
		self.char_list = char_list
		self.space = sym_space
		self.blank = sym_blank
		self.report_cer = report_cer
		self.report_wer = report_wer
		self.idx_blank = self.char_list.index(self.blank)
		if self.space in self.char_list:
			self.idx_space = self.char_list.index(self.space)
		else:
			self.idx_space = None

	def __call__(self, ys_hat, ys_pad, is_ctc=False):
		cer, wer = None, None
		if is_ctc:
			return self.calculate_cer_ctc(ys_hat, ys_pad)
		elif not self.report_cer and not self.report_wer:
			return cer, wer

		seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
		if self.report_cer:
			cer = self.calculate_cer(seqs_hat, seqs_true)

		if self.report_wer:
			wer = self.calculate_wer(seqs_hat, seqs_true)
		return cer, wer

	def calculate_cer_ctc(self, ys_hat, ys_pad):
		cers, char_ref_lens = [], []
		for i, y in enumerate(ys_hat):
			y_hat = [x[0] for x in groupby(y)]
			y_true = ys_pad[i]
			seq_hat, seq_true = [], []
			for idx in y_hat:
				idx = int(idx)
				if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
					seq_hat.append(self.char_list[int(idx)])

			for idx in y_true:
				idx = int(idx)
				if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
					seq_true.append(self.char_list[int(idx)])

			hyp_chars = "".join(seq_hat)
			ref_chars = "".join(seq_true)
			if len(ref_chars) > 0:
				cers.append(editdistance.eval(hyp_chars, ref_chars))
				char_ref_lens.append(len(ref_chars))

		cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
		return cer_ctc

	def convert_to_char(self, ys_hat, ys_pad):
		seqs_hat, seqs_true = [], []
		for i, y_hat in enumerate(ys_hat):
			y_true = ys_pad[i]
			eos_true = np.where(y_true == -1)[0]
			eos_true = eos_true[0] if len(eos_true) > 0 else len(y_true)
			# To avoid wrong higger WER than the one obtained from the decoding
			# eos from y_true is used to mark the eos in y_hat
			# because of that y_hats has not padded outs with -1.
			seq_hat = [self.char_list[int(idx)] for idx in y_hat[:eos_true]]
			seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
			seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
			seq_hat_text = seq_hat_text.replace(self.blank, '')
			seq_true_text = "".join(seq_true).replace(self.space, ' ')
			seqs_hat.append(seq_hat_text)
			seqs_true.append(seq_true_text)
		return seqs_hat, seqs_true

	def calculate_cer(self, seqs_hat, seqs_true):
		char_eds, char_ref_lens = [], []
		for i, seq_hat_text in enumerate(seqs_hat):
			seq_true_text = seqs_true[i]
			hyp_chars = seq_hat_text.replace(' ', '')
			ref_chars = seq_true_text.replace(' ', '')
			char_eds.append(editdistance.eval(hyp_chars, ref_chars))
			char_ref_lens.append(len(ref_chars))
		return float(sum(char_eds)) / sum(char_ref_lens)

	def calculate_wer(self, seqs_hat, seqs_true):
		word_eds, word_ref_lens = [], []
		for i, seq_hat_text in enumerate(seqs_hat):
			seq_true_text = seqs_true[i]
			hyp_words = seq_hat_text.split()
			ref_words = seq_true_text.split()
			word_eds.append(editdistance.eval(hyp_words, ref_words))
			word_ref_lens.append(len(ref_words))
		return float(sum(word_eds)) / sum(word_ref_lens)


class PreProcess(nn.Module):

	def __init__(self):
		super(PreProcess, self).__init__()

	def forward(self, data):

		# ToDo - write the code for generating pitch features

		pre_emphasis = config.fbank['pre_emphasis']
		frame_size = config.fbank['frame_size']
		frame_stride = config.fbank['frame_stride']
		n_fft = config.fbank['n_fft']
		rate = config.fbank['rate']

		emphasized_data = torch.zeros_like(data).float()

		if config.use_cuda:
			emphasized_data = emphasized_data.cuda()

		emphasized_data[:, 1:] = data[:, 1:] - pre_emphasis * data[:, :-1]
		emphasized_data[:, 0] = data[:, 0]

		frame_length, frame_step = frame_size * rate, frame_stride * rate  # Convert from seconds to samples
		frame_length = int(frame_length)
		frame_step = int(frame_step)

		mag_frames = torch.norm(
			torch.stft(
				emphasized_data,
				n_fft=n_fft,
				hop_length=frame_step,
				win_length=frame_length,
				window=torch.hamming_window(frame_length),
				pad_mode='constant'
			), dim=3).transpose(2, 1)

		pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))  # Power Spectrum

		filter_banks = torch.matmul(pow_frames, fbank.transpose(1, 0))
		filter_banks[filter_banks == 0] = 2.220446049250313e-16
		filter_banks = 20 * torch.log10(filter_banks)  # dB
		filter_banks -= (torch.mean(filter_banks, dim=(0, 1), keepdim=True) + 1e-8)

		return filter_banks, (torch.ones(filter_banks.shape[0])*filter_banks.shape[1]).long()


class E2E(ASRInterface, torch.nn.Module):

	@staticmethod
	def add_arguments(parser):
		group = parser.add_argument_group("transformer model setting")
		group.add_argument(
			"--transformer-init",
			type=str,
			default="pytorch",
			choices=["pytorch", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"],
			help='how to initialize transformer parameters'
		)
		group.add_argument(
			"--transformer-input-layer", type=str, default="conv2d", choices=["conv2d", "linear", "embed"],
			help='transformer input layer type'
		)
		group.add_argument(
			'--transformer-attn-dropout-rate', default=None, type=float,
			help='dropout in transformer attention. use --dropout-rate if None is set')
		group.add_argument('--transformer-lr', default=10.0, type=float, help='Initial value of learning rate')
		group.add_argument('--transformer-warmup-steps', default=25000, type=int, help='optimizer warmup steps')
		group.add_argument(
			'--transformer-length-normalized-loss', default=True, type=strtobool, help='normalize loss by length')

		return parser

	def __init__(self, idim, odim, args, ignore_id=-1, char_list=None):
		torch.nn.Module.__init__(self)

		self.pre_process = PreProcess()

		if args.transformer_attn_dropout_rate is None:
			args.transformer_attn_dropout_rate = args.dropout_rate
		self.encoder = Encoder(
			idim=idim,
			attention_dim=args.adim,
			attention_heads=args.aheads,
			linear_units=args.eunits,
			num_blocks=args.elayers,
			input_layer=args.transformer_input_layer,
			dropout_rate=args.dropout_rate,
			positional_dropout_rate=args.dropout_rate,
			attention_dropout_rate=args.transformer_attn_dropout_rate
		)
		self.decoder = Decoder(
			odim=odim,
			attention_dim=args.adim,
			attention_heads=args.aheads,
			linear_units=args.dunits,
			num_blocks=args.dlayers,
			dropout_rate=args.dropout_rate,
			positional_dropout_rate=args.dropout_rate,
			self_attention_dropout_rate=args.transformer_attn_dropout_rate,
			src_attention_dropout_rate=args.transformer_attn_dropout_rate
		)
		self.sos = odim - 1
		self.eos = odim - 1
		self.odim = odim
		self.ignore_id = ignore_id
		self.subsample = [1]

		# self.lsm_weight = a
		self.criterion = LabelSmoothingLoss(
			self.odim, self.ignore_id, args.lsm_weight, args.transformer_length_normalized_loss)
		# self.verbose = args.verbose
		self.reset_parameters(args)
		self.adim = args.adim
		self.mtlalpha = args.mtlalpha
		if args.mtlalpha > 0.0:
			self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
		else:
			self.ctc = None

		if args.report_cer or args.report_wer:
			self.error_calculator = ErrorCalculator(
				char_list,
				args.sym_space, args.sym_blank,
				args.report_cer, args.report_wer
			)
		else:
			self.error_calculator = None
		self.rnnlm = None

	def reset_parameters(self, args):
		if args.transformer_init == "pytorch":
			return
		# weight init
		for p in self.parameters():
			if p.dim() > 1:
				if args.transformer_init == "xavier_uniform":
					torch.nn.init.xavier_uniform_(p.data)
				elif args.transformer_init == "xavier_normal":
					torch.nn.init.xavier_normal_(p.data)
				elif args.transformer_init == "kaiming_uniform":
					torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
				elif args.transformer_init == "kaiming_normal":
					torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
				else:
					raise ValueError("Unknown initialization: " + args.transformer_init)
		# bias init
		for p in self.parameters():
			if p.dim() == 1:
				p.data.zero_()
		# reset some modules with default init
		for m in self.modules():
			if isinstance(m, (torch.nn.Embedding, LayerNorm)):
				m.reset_parameters()

	def add_sos_eos(self, ys_pad):
		from pytorch_backend.nets_utils import pad_list
		eos = ys_pad.new([self.eos])
		sos = ys_pad.new([self.sos])
		ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
		ys_in = [torch.cat([sos, y], dim=0) for y in ys]
		ys_out = [torch.cat([y, eos], dim=0) for y in ys]
		return pad_list(ys_in, self.eos), pad_list(ys_out, self.ignore_id)

	def target_mask(self, ys_in_pad):
		ys_mask = ys_in_pad != self.ignore_id
		m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
		return ys_mask.unsqueeze(-2) & m

	def forward(self, audio, audio_length, ys_pad):

		# audio = [audio[i][0:audio_length[i]] for i in range(audio.shape[0])]

		xs_pad, ilens = self.pre_process(audio)

		if config.use_cuda:
			xs_pad = xs_pad.cuda()

		'''E2E forward

		:param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
		:param torch.Tensor ilens: batch of lengths of source sequences (B)
		:param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
		:return: ctc loass value
		:rtype: torch.Tensor
		:return: attention loss value
		:rtype: torch.Tensor
		:return: accuracy in attention decoder
		:rtype: float
		'''

		# forward encoder
		xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
		src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
		hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
		self.hs_pad = hs_pad

		# forward decoder
		ys_in_pad, ys_out_pad = self.add_sos_eos(ys_pad)
		ys_mask = self.target_mask(ys_in_pad)
		pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
		self.pred_pad = pred_pad

		# compute loss
		loss_att = self.criterion(pred_pad, ys_out_pad)
		self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)

		# TODO(karita) show predected text
		# TODO(karita) calculate these stats
		cer_ctc = None
		if self.mtlalpha == 0.0:
			loss_ctc = None
		else:
			batch_size = xs_pad.size(0)
			hs_len = hs_mask.view(batch_size, -1).sum(1)
			loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
			if self.error_calculator is not None:
				ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
				cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

		# copyied from e2e_asr
		alpha = self.mtlalpha
		if alpha == 0:
			self.loss = loss_att
		elif alpha == 1:
			self.loss = loss_ctc
		else:
			self.loss = alpha * loss_ctc + (1 - alpha) * loss_att

		# 5. compute cer/wer
		if self.training or self.error_calculator is None:
			return self.loss, loss_att, loss_ctc
		else:
			ys_hat = pred_pad.argmax(dim=-1)
			cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
			return self.loss, loss_att, loss_ctc, cer, wer, ys_hat, ys_pad

	def recognize(self, feat, recog_args, char_list=None, rnnlm=None, use_jit=False):
		"""recognize feat

		:param ndnarray x: input acouctic feature (B, T, D) or (T, D)
		:param namespace recog_args: argment namespace contraining options
		:param list char_list: list of characters
		:param torch.nn.Module rnnlm: language model module
		:return: N-best decoding results
		:rtype: list

		TODO(karita): do not recompute previous attention for faster decoding
		"""
		self.eval()
		feat = torch.as_tensor(feat).unsqueeze(0)
		enc_output, _ = self.encoder(feat, None)
		# return enc_output.data.cpu().numpy()[0]

		if recog_args.ctc_weight > 0.0:
			lpz = self.ctc.log_softmax(enc_output)
			lpz = lpz.squeeze(0)
		else:
			lpz = None

		h = enc_output.squeeze(0)

		print('input lengths: ' + str(h.size(0)))
		# search parms
		beam = recog_args.beam_size
		penalty = recog_args.penalty
		ctc_weight = recog_args.ctc_weight

		# preprare sos
		y = self.sos
		vy = h.new_zeros(1).long()

		if recog_args.maxlenratio == 0:
			maxlen = h.shape[0]
		else:
			# maxlen >= 1
			maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
		minlen = int(recog_args.minlenratio * h.size(0))
		print('max output length: ' + str(maxlen))
		print('min output length: ' + str(minlen))

		# initialize hypothesis
		if rnnlm:
			hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
		else:
			hyp = {'score': 0.0, 'yseq': [y]}
		if lpz is not None:

			ctc_prefix_score = CTCPrefixScore(lpz.data.cpu().numpy(), 0, self.eos, np)
			hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
			hyp['ctc_score_prev'] = 0.0
			if ctc_weight != 1.0:
				# pre-pruning based on attention scores
				from pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
				ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
			else:
				ctc_beam = lpz.shape[-1]
		hyps = [hyp]
		ended_hyps = []

		traced_decoder = None
		for i in range(maxlen):
			print('position ' + str(i))

			hyps_best_kept = []
			for hyp in hyps:
				vy.unsqueeze(1)
				vy[0] = hyp['yseq'][i]

				# get nbest local scores and their ids
				ys_mask = subsequent_mask(i + 1).unsqueeze(0)
				ys = torch.tensor(hyp['yseq']).unsqueeze(0)
				# FIXME: jit does not match non-jit result
				if use_jit:
					if traced_decoder is None:
						traced_decoder = torch.jit.trace(self.decoder.recognize, (ys, ys_mask, enc_output))
					local_att_scores = traced_decoder(ys, ys_mask, enc_output)
				else:
					local_att_scores = self.decoder.recognize(ys, ys_mask, enc_output)

				if rnnlm:
					rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
					local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
				else:
					local_scores = local_att_scores

				if lpz is not None:
					local_best_scores, local_best_ids = torch.topk(
						local_att_scores, ctc_beam, dim=1)
					ctc_scores, ctc_states = ctc_prefix_score(
						hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])

					local_scores = \
						(1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
						+ ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev']).cuda()
					if rnnlm:
						local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
					local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
					local_best_ids = local_best_ids[:, joint_best_ids[0]]
				else:
					local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

				for j in range(beam):
					new_hyp = dict()
					new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
					new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
					new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
					new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
					if rnnlm:
						new_hyp['rnnlm_prev'] = rnnlm_state
					if lpz is not None:
						new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
						new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
					# will be (2 x beam) hyps at most
					hyps_best_kept.append(new_hyp)

				hyps_best_kept = sorted(
					hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

			# sort and get nbest
			hyps = hyps_best_kept
			print('number of pruned hypothes: ' + str(len(hyps)))
			if char_list is not None:
				print(
					'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

			# add eos in the final loop to avoid that there are no ended hyps
			if i == maxlen - 1:
				print('adding <eos> in the last postion in the loop')
				for hyp in hyps:
					hyp['yseq'].append(self.eos)

			# add ended hypothes to a final list, and removed them from current hypothes
			# (this will be a probmlem, number of hyps < beam)
			remained_hyps = []
			for hyp in hyps:
				if hyp['yseq'][-1] == self.eos:
					# only store the sequence that has more than minlen outputs
					# also add penalty
					if len(hyp['yseq']) > minlen:
						hyp['score'] += (i + 1) * penalty
						if rnnlm:  # Word LM needs to add final <eos> score
							hyp['score'] += recog_args.lm_weight * rnnlm.final(
								hyp['rnnlm_prev'])
						ended_hyps.append(hyp)
				else:
					remained_hyps.append(hyp)

			# end detection
			if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
				print('end detected at %d', i)
				break

			hyps = remained_hyps
			if len(hyps) > 0:
				print('remeined hypothes: ' + str(len(hyps)))
			else:
				print('no hypothesis. Finish decoding.')
				break

			if char_list is not None:
				for hyp in hyps:
					print(
						'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

			print('number of ended hypothes: ' + str(len(ended_hyps)))

		nbest_hyps = sorted(
			ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

		# check number of hypotheis
		if len(nbest_hyps) == 0:
			print('there is no N-best results, perform recognition again with smaller minlenratio.')
			# should copy becasuse Namespace will be overwritten globally
			recog_args = Namespace(**vars(recog_args))
			recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
			return self.recognize(feat, recog_args, char_list, rnnlm)

		print('total log probability: ' + str(nbest_hyps[0]['score']))
		print('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
		return nbest_hyps

	def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
		'''E2E attention calculation

		:param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
		:param torch.Tensor ilens: batch of lengths of input sequences (B)
		:param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
		:return: attention weights with the following shape,
			1) multi-head case => attention weights (B, H, Lmax, Tmax),
			2) other case => attention weights (B, Lmax, Tmax).
		:rtype: float ndarray
		'''
		with torch.no_grad():
			self.forward(xs_pad, ilens, ys_pad)
		ret = dict()
		for name, m in self.named_modules():
			if isinstance(m, MultiHeadedAttention):
				ret[name] = m.attn.cpu().numpy()
		return ret


class Model(nn.Module):

	def __init__(self):
		super(Model, self).__init__()

	def forward(self, x):
		return x
