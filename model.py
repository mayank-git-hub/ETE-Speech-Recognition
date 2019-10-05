import torch
from torch import nn
import config
import math
import numpy as np


class PreProcess(nn.Module):

	def __init__(self):

		super(PreProcess, self).__init__()

	def forward(self, data):

		# ToDo - write unit test function for calculate_fbank
		# ToDo - write the code for generating pitch features

		pre_emphasis = config.fbank['pre_emphasis']
		frame_size = config.fbank['frame_size']
		frame_stride = config.fbank['frame_stride']
		n_fft = config.fbank['n_fft']
		nfilt = config.fbank['nfilt']
		rate = config.fbank['rate']

		data = torch.from_numpy(data)

		emphasized_data = torch.zeros_like(data)
		emphasized_data[1:] = data[1:] - pre_emphasis * data[:-1]
		emphasized_data[0] = data[0]

		frame_length, frame_step = frame_size * rate, frame_stride * rate  # Convert from seconds to samples
		frame_length = int(round(frame_length))
		frame_step = int(round(frame_step))
		num_frames = int(
			math.ceil(abs(data.shape[0] - frame_length)) / frame_step)  # Make sure that we have at least 1 frame

		pad_signal_length = num_frames * frame_step + frame_length
		z = torch.zeros((pad_signal_length - data.shape[0]))
		pad_signal = torch.stack((emphasized_data, z), dim=0)

		indices = \
			torch.arange(frame_length).repeat(num_frames, 1) + \
			torch.arange(0, num_frames * frame_step, frame_step).repeat(frame_length, 1).transpose(1, 0)

		frames = pad_signal[indices.long()]
		frames *= torch.hamming_window(frame_length)

		# ToDo - shifting back to torch because of no parameters to change n_fft

		frames = frames.numpy()
		mag_frames = np.absolute(np.fft.rfft(frames, n_fft))  # Magnitude of the FFT
		pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))  # Power Spectrum

		# Shifting back to torch

		pow_frames = torch.from_numpy(pow_frames)

		#############################################################################

		low_freq_mel = 0
		high_freq_mel = (2595 * math.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
		mel_points = torch.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
		hz_points = (700 * (torch.pow(mel_points / 2595, 10) - 1))  # Convert Mel to Hz
		bin_ = torch.floor((n_fft + 1) * hz_points / rate)

		fbank = torch.zeros((nfilt, int(torch.floor(n_fft / 2 + 1))))

		# ToDo - Check if this for loop can be removed

		for m in range(1, nfilt + 1):
			f_m_minus = int(bin_[m - 1])  # left
			f_m = int(bin_[m])  # center
			f_m_plus = int(bin_[m + 1])  # right

			# ToDo - Check if this arange works as expected

			fbank[m - 1, torch.arange(f_m_minus, f_m)] = (torch.arange(f_m_minus, f_m) - bin_[m - 1]) / (
						bin_[m] - bin_[m - 1])
			fbank[m - 1, torch.arange(f_m, f_m_plus)] = (bin_[m + 1] - torch.arange(f_m, f_m_plus)) / (
						bin_[m + 1] - bin_[m])

		filter_banks = torch.dot(pow_frames, fbank.transpose(1, 0))
		filter_banks = torch.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
		filter_banks = 20 * torch.log10(filter_banks)  # dB

		filter_banks -= (torch.mean(filter_banks, dim=0) + 1e-8)

		return filter_banks


class Model(nn.Module):

	def __init__(self):

		super(Model, self).__init__()

		self.pre_process = PreProcess()

	def forward(self, x):

		x = self.pre_process(x)

		return x
