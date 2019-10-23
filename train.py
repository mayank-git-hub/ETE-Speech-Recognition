from dataloader import DataLoaderTrain
from torch.utils import data
import config
from model import E2E
from pytorch_backend.transformer.optimizer import get_std_opt


def train():

	pass


def main():

	dataloader = data.DataLoader(
		DataLoaderTrain(),
		batch_size=config.train_param['batch_size'],
		num_workers=config.train_param['num_workers']
	)

	model = E2E(idim=80, odim=5002, args=config.ModelArgs())
	optimizer = get_std_opt(
		model, config.train_param['adim'], config.train_param['transformer_warmup_steps'], config.train_param['lr'])

	"""
		Dataloader should produce 
			1) audio array
			2) path to audio array
			3) Token transcript
			4) Word transcript
			
		After dataloader gives the output, all the ys should be padded to get the same size, (pad with -1)
	"""
	"""
	"""
	# ToDo - Implement Custom Updater
	# ToDo - Implement dataloader
	# ToDo - Implement trainer
	# ToDo - Implement tester


if __name__ == "__main__":

	main()
