from dataloader import DataLoaderTrain, my_collate
from torch.utils import data
import config
from model import E2E
from pytorch_backend.transformer.optimizer import get_std_opt
from tqdm import tqdm
import torch
import numpy as np
from datetime import datetime
from torch import nn
from unigram_gen import create_unigram_model, create_json
import os
import time


def train(epoch_start, model, optimizer):

	# ToDo - Create tests to check whether the model is training
	# ToDo - Visualize the outputs
	# ToDo - Visualize the attention outputs
	# ToDo - Create tests to check that the Fbanks are being generated correctly

	for epoch_i in range(epoch_start, config.num_epochs):

		all_loss = []
		all_loss_ctc = []
		all_loss_att = []

		running_loss = 0
		running_loss_ctc = 0
		running_loss_att = 0

		dataloader = tqdm(data.DataLoader(
			DataLoaderTrain(),
			batch_size=config.train_param['batch_size'],
			num_workers=config.train_param['num_workers'],
			collate_fn=my_collate,
			shuffle=True
		))

		optimizer.zero_grad()
		model.train()

		prev_lr = 0

		for no, (audio, audio_length, path, text, token, token_id) in enumerate(dataloader):

			if audio.shape[1] >= 270000:
				continue

			if config.use_cuda:

				audio = audio.cuda()
				audio_length = audio_length.cuda()
				token_id = token_id.cuda()

			loss, loss_att, loss_ctc = model(audio, audio_length, token_id)

			loss = loss.mean()
			loss_att = loss_att.mean()
			loss_ctc = loss_ctc.mean()

			loss.backward()

			if (no + 1) % config.train_param['accum_grad'] == 0:
				optimizer.step()
				optimizer.zero_grad()

			all_loss.append(loss.item())
			all_loss_att.append(loss_att.item())
			all_loss_ctc.append(loss_ctc.item())

			running_loss = (running_loss*no + loss.item())/(no + 1)
			running_loss_ctc = (running_loss_ctc*no + loss_ctc.item())/(no + 1)
			running_loss_att = (running_loss_att*no + loss_att.item())/(no + 1)

			dataloader.set_description(
				'Epoch: {6} | '
				'LR: {7:.6f} | '
				'Loss: {0:.3f} | '
				'Avg. Loss: {3:.3f} | '
				'Loss_Att: {1:.3f} | '
				'Avg Loss_Att: {4:.3f} | '
				'Loss_CTC: {2:.3f} | '
				'Avg Loss_CTC: {5:.3f}'.format(
					loss.item(),
					loss_att.item(),
					loss_ctc.item(),
					running_loss,
					running_loss_att,
					running_loss_ctc,
					epoch_i,
					optimizer._rate
				)
			)

		optimizer.zero_grad()

		cur_time = datetime.time(datetime.now())

		with open(config.model_save_path + '/LibriSpeech_train960.{0}.{1:.4f}.{2}.pth'.format(
					epoch_i, np.mean(all_loss), cur_time), 'wb') as f:

			torch.save({
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch_i,
				'Losses': [all_loss, all_loss_att, all_loss_ctc],
				'datetime': str(datetime.time(datetime.now()))
			}, f)


def main():

	create_unigram_model()
	create_json()

	model = E2E(idim=80, odim=5002, args=config.ModelArgs())

	if config.use_cuda:
		model = model.cuda()
		model = nn.DataParallel(model)

	optimizer = get_std_opt(
		model, config.train_param['adim'], config.train_param['transformer_warmup_steps'], config.train_param['lr'])

	if config.resume['restart']:
		checkpoint = torch.load(config.resume['model_path'])
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch_start = checkpoint['epoch'] + 1
		losses = checkpoint['Losses']
		print(
			'Loss for the epoch:', epoch_start, 
			' | Avg. Loss: {0:.4f} | '
			'Avg Loss_Att: {1:.4f} | '
			'Avg Loss_CTC: {2:.4f}'.format(np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])))
	else:
		epoch_start = 0

	train(epoch_start, model, optimizer)


if __name__ == "__main__":

	main()
