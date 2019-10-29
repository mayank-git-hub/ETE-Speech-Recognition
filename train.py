from dataloader import DataLoaderTrain, my_collate
from torch.utils import data
import config
from model import E2E
from pytorch_backend.transformer.optimizer import get_std_opt


def train(model, optimizer):

	# ToDo - Create tests to check whether the model is training
	# ToDo - Visualize the outputs
	# ToDo - Visualize the attention outputs
	# ToDo - Create tests to check that the Fbanks are being generated correctly

	for epoch_i in range(config.num_epochs):

		dataloader = data.DataLoader(
			DataLoaderTrain(),
			batch_size=config.train_param['batch_size'],
			num_workers=config.train_param['num_workers'],
			collate_fn=my_collate
		)

		optimizer.zero_grad()
		model.train()

		for (audio, path, text, token, token_id) in dataloader:

			if config.use_cuda:

				audio = [audio_i.cuda() for audio_i in audio]
				token_id = token_id.cuda()

			loss = model(audio, token_id)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()


def main():

	model = E2E(idim=80, odim=5002, args=config.ModelArgs())
	if config.use_cuda:
		model = model.cuda()
	optimizer = get_std_opt(
		model, config.train_param['adim'], config.train_param['transformer_warmup_steps'], config.train_param['lr'])

	train(model, optimizer)

	# Implement Custom Updater
	# Implement dataloader
	# Implement trainer
	# ToDo - Implement tester


if __name__ == "__main__":

	main()
