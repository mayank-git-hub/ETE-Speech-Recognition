from torch.utils import data
import config
import os
import pickle
import soundfile as sf
from tinytag import TinyTag
from tqdm import tqdm


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


def get_audio_from_data(all_data, item):

	speaker_id, chapter_id, sentence_id, trans, path, gender = all_data[item]

	audio, rate = sf.read(path)

	return audio


class DataLoaderTrain(data.Dataset):

	def __init__(self):

		self.all_data = load_filename(config.train_set, 'DataLoaderTrain.cache')

	def __getitem__(self, item):

		return get_audio_from_data(self.all_data, item)

	def __len__(self):

		return len(self.all_data)


class DataLoaderRecog(data.Dataset):

	def __init__(self):

		self.all_data = load_filename(config.recog_set, 'DataLoaderRecog.cache')

	def __getitem__(self, item):

		return get_audio_from_data(self.all_data, item)

	def __len__(self):

		return len(self.all_data)


class DataLoaderDev(data.Dataset):

	def __init__(self):

		self.all_data = load_filename(config.train_dev, 'DataLoaderDev.cache')

	def __getitem__(self, item):

		return get_audio_from_data(self.all_data, item)

	def __len__(self):

		return len(self.all_data)


if __name__ == "__main__":

	DataLoaderTrain()
	DataLoaderRecog()
	DataLoaderDev()
