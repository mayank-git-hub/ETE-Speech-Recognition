import sentencepiece as sp
import os
from dataloader import DataLoaderTrain, DataLoaderDev, DataLoaderRecog
from tqdm import tqdm
import json
import config


def create_unigram_model():

	os.makedirs(config.cache_dir + '/json_data', exist_ok=True)

	all_data = DataLoaderTrain(only_all_data=True).all_data

	os.makedirs(config.cache_dir + '/unigram_model', exist_ok=True)

	if not os.path.exists(config.cache_dir + '/unigram_model/All_Transcript.txt'):
		print('Creating All Transcripts')
		with open(config.cache_dir + '/unigram_model/All_Transcript.txt', 'w') as writer:
			for sentence in tqdm(all_data):
				writer.write(sentence[3]+'\n')

	if not (os.path.exists(config.cache_dir + '/unigram_model/unigram.model') and os.path.exists(config.cache_dir + '/unigram_model/unigram.vocab')):
		sp.SentencePieceTrainer.Train(
			'--input=' + config.cache_dir + '/unigram_model/All_Transcript.txt '
			'--vocab_size=5000 '
			'--model_type=unigram '
			'--model_prefix=' + config.cache_dir + '/unigram_model/unigram '
			'--input_sentence_size=100000000'
		)

	if not os.path.exists(config.cache_dir + '/unigram_model/unigram_units.txt'):
		spm = sp.SentencePieceProcessor()
		spm.load(config.cache_dir + '/unigram_model/unigram.model')

		encoded_output = set()

		with open(config.cache_dir + '/unigram_model/All_Transcript.txt', 'r') as f:
			print('Reading the transcript')
			all_text = []
			for i in f:
				all_text.append(i[:-1])
			for no, i in enumerate(tqdm(all_text)):
				encoded_output = encoded_output.union(set(spm.encode_as_pieces(i)))

		print('First done')
		encoded_output = sorted(list(set(encoded_output)))
		with open(config.cache_dir + '/unigram_model/unigram_units.txt', 'w') as f:
			print('Writing the encoded units output')
			f.write('<unk> 1\n')
			for no, encoded in enumerate(tqdm(encoded_output)):
				f.write(encoded + ' ' + str(no + 2) + '\n')


def create_json_(all_data, spm, piece_to_id):

	json_ = {
		'utts': {}
	}

	for all_data_i in tqdm(all_data):
		speaker_id, chapter_id, sentence_id, trans, path_flac, gender = all_data_i
		pieces = spm.encode_as_pieces(trans)
		piece_id = [piece_to_id[_] for _ in pieces]
		json_['utts'][speaker_id + '-' + chapter_id + '-' + sentence_id] = {
			'input': [
				{
					"path_flac": path_flac,
				}
			],
			'output': [
				{
					"text": trans,
					"token": ' '.join(pieces),
					"tokenid": ' '.join(piece_id)
				}
			]
		}

	return json_


def create_json():

	os.makedirs(config.cache_dir + '/json_data', exist_ok=True)

	spm = sp.SentencePieceProcessor()
	spm.load(config.cache_dir + '/unigram_model/unigram.model')
	piece_to_id = {}

	with open(config.cache_dir + '/unigram_model/unigram_units.txt', 'r') as f:
		for i in f:
			piece_to_id[i.split()[0]] = i.split()[1]

	if not os.path.exists(config.cache_dir + '/json_data/recog.json'):
		print('Recog json')
		recog_json = create_json_(DataLoaderRecog(only_all_data=True).all_data, spm, piece_to_id)

		with open(config.cache_dir + '/json_data/recog.json', 'w', encoding='utf8') as f:

			json.dump(recog_json, f, indent=4, separators=(',', ': '), sort_keys=True, ensure_ascii=False)

	if not os.path.exists(config.cache_dir + '/json_data/dev.json'):
		print('Dev json')
		dev_json = create_json_(DataLoaderDev(only_all_data=True).all_data, spm, piece_to_id)

		with open(config.cache_dir + '/json_data/dev.json', 'w', encoding='utf8') as f:

			json.dump(dev_json, f, indent=4, separators=(',', ': '), sort_keys=True, ensure_ascii=False)

	if not os.path.exists(config.cache_dir + '/json_data/train.json'):
		print('Train json')
		train_json = create_json_(DataLoaderTrain(only_all_data=True).all_data, spm, piece_to_id)

		with open(config.cache_dir + '/json_data/train.json', 'w', encoding='utf8') as f:

			json.dump(train_json, f, indent=4, separators=(',', ': '), sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":

	create_unigram_model()
	create_json()
