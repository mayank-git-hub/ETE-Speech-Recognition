from torch.utils import data
import config
import os
import pickle
import soundfile as sf
from tinytag import TinyTag
from tqdm import tqdm
import torch
import json
import numpy as np
import torch.nn.functional as F


def load_filename(dataset, cache_name):

	if not os.path.exists(config.cache_dir + '/' + cache_name):

		all_data = {}
		gender_mapping = {}
		length = 0

		with open(config.path_to_download + '/SPEAKERS.TXT', 'r') as f:
			for i in f:
				if i[0] == ';':
					continue

				i = i.split('|')

				gender_mapping[i[0].lstrip().rstrip()] = i[1].lstrip().rstrip()

		for set_i in tqdm(dataset):

			path_1 = config.path_to_download + '/' + set_i

			for speaker_i in tqdm(sorted(os.listdir(path_1))):

				if speaker_i == '.complete':
					continue

				all_data[speaker_i] = {}
				gender = gender_mapping[speaker_i]

				path_2 = path_1 + '/' + speaker_i

				for chapter_i in sorted(os.listdir(path_2)):

					all_data[speaker_i][chapter_i] = {}

					path_3 = path_2 + '/' + chapter_i

					trans_path = path_3 + '/' + speaker_i + '-' + chapter_i + '.trans.txt'

					with open(trans_path, 'r') as trans:

						for trans_i in trans:
							trans_i = trans_i.split()
							sentence_i = (trans_i[0]).split('-')[2]
							transcript = ' '.join(trans_i[1:])

							# Remove max-min characters
							if config.rmlsutt['min_chars'] < len(transcript.replace(' ', '')) < config.rmlsutt['max_chars']:

								flac_path = path_3 + '/' + speaker_i + '-' + chapter_i + '-' + sentence_i + '.flac'
								tag = TinyTag.get(flac_path)
								num_samples = tag.duration*tag.samplerate
								sample_rate = tag.samplerate

								assert sample_rate == config.fbank['rate'], 'Rate is different than in fbank'

								frame_size = int(config.fbank['frame_size']*sample_rate)
								frame_stride = int(config.fbank['frame_stride']*sample_rate)

								num_frames = (num_samples - frame_size)//frame_stride

								# Remove max-min frames
								if config.rmlsutt['min_frames'] < num_frames < config.rmlsutt['max_frames']:

									all_data[speaker_i][chapter_i][sentence_i] = {
										'trans': transcript,
										'path': flac_path,
										'gender': gender,
									}

									length += 1

		with open(config.cache_dir + '/' + cache_name, 'wb') as f:
			pickle.dump([length, all_data], f)

	else:

		with open(config.cache_dir + '/' + cache_name, 'rb') as f:
			length, all_data = pickle.load(f)

	all_flat = []

	for speaker_id in all_data:
		for chapter_id in all_data[speaker_id]:
			for sentence_id in all_data[speaker_id][chapter_id]:
				values = all_data[speaker_id][chapter_id][sentence_id]
				all_flat.append([speaker_id, chapter_id, sentence_id, values['trans'], values['path'], values['gender']])

	assert length == len(all_flat), 'Length not equal to len(all_flat)'

	return all_flat


def get_audio_from_data(all_data, all_meta, item):

	speaker_id, chapter_id, sentence_id, trans, path, gender = all_data[item]

	target_ = all_meta['utts'][str(speaker_id) + '-' + str(chapter_id) + '-' + str(sentence_id)]['output'][0]

	audio, rate = sf.read(path)

	return audio, path, target_['text'], target_['token'], np.array(target_['tokenid'].split()).astype(np.int32)


def my_collate(batch):

	all_audio = []
	all_path = []
	all_text = []
	all_token = []
	all_token_id = []

	max_y = 0

	for (audio, path, text, token, token_id) in batch:

		all_audio.append(torch.from_numpy(audio).float())
		all_path.append(path)
		all_text.append(text)
		all_token.append(token)
		all_token_id.append(torch.from_numpy(token_id).long())
		max_y = max(max_y, token_id.shape[0])

	for all_token_id_i in range(len(all_token_id)):

		all_token_id[all_token_id_i] = F.pad(
			all_token_id[all_token_id_i],
			[0, max_y - all_token_id[all_token_id_i].shape[0]],
			mode='constant',
			value=-1
		).unsqueeze(0)

	all_token_id = torch.cat(all_token_id, dim=0)

	return all_audio, all_path, all_text, all_token, all_token_id


class DataLoaderTrain(data.Dataset):

	def __init__(self, only_all_data=False):

		self.all_data = load_filename(config.train_set, 'DataLoaderTrain.cache')
		if only_all_data:
			return
		with open('Cache/json_data/train.json', 'r') as f:
			self.all_meta = json.load(f)

	def __getitem__(self, item):

		return get_audio_from_data(self.all_data, self.all_meta, item)

	def __len__(self):

		return len(self.all_data)


class DataLoaderRecog(data.Dataset):

	def __init__(self):

		self.all_data = load_filename(config.recog_set, 'DataLoaderRecog.cache')
		with open('Cache/json_data/recog.json', 'r') as f:
			self.all_meta = json.load(f)

	def __getitem__(self, item):

		return get_audio_from_data(self.all_data, self.all_meta, item)

	def __len__(self):

		return len(self.all_data)


class DataLoaderDev(data.Dataset):

	def __init__(self):

		self.all_data = load_filename(config.train_dev, 'DataLoaderDev.cache')
		with open('Cache/json_data/dev.json', 'r') as f:
			self.all_meta = json.load(f)

	def __getitem__(self, item):

		return get_audio_from_data(self.all_data, self.all_meta, item)

	def __len__(self):

		return len(self.all_data)


if __name__ == "__main__":
	"""
		Dataloader should produce 
			1) audio array
			2) path to audio array
			3) Token transcript
			4) Token ID transcript
			5) Word transcript

		After dataloader gives the output, all the ys should be padded to get the same size, (pad with -1)
	"""

	from torch.utils.data import DataLoader

	dev_loader = DataLoader(DataLoaderDev(), batch_size=5, num_workers=5, collate_fn=my_collate)
	train_loader = DataLoader(DataLoaderTrain(), batch_size=5, num_workers=5, collate_fn=my_collate)
	recog_loader = DataLoader(DataLoaderRecog(), batch_size=5, num_workers=5, collate_fn=my_collate)
