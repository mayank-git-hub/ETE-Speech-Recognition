from dataloader import DataLoaderTrain
from torch.utils import data
import config
from model import E2E
from torch.optim import Adam


def train():

	pass


def main():

	dataloader = data.DataLoader(
		DataLoaderTrain(),
		batch_size=config.train_param['batch_size'],
		num_workers=config.train_param['num_workers']
	)

	model = E2E()
	optimizer = Adam(model.parameters(), config.train_param['lr'])


if __name__ == "__main__":

	main()
