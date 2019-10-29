import click
import torch
import numpy as np
import random


def seed(config=None):

	# This removes randomness, makes everything deterministic

	if config is None:
		import config

	np.random.seed(config.seed)
	random.seed(config.seed)
	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.backends.cudnn.deterministic = True


@click.group()
def main():
	seed()
	pass


@main.command()
def train():

	import train

	print('Starting Training')
	train.main()


@main.command()
def test():

	import test

	print('Starting Testing')
	test.main()


@main.command()
def pre_process():

	import dataloader
	import unigram_gen

	dataloader.DataLoaderTrain()
	dataloader.DataLoaderRecog()
	dataloader.DataLoaderDev()

	unigram_gen.create_unigram_model()
	unigram_gen.create_json()


if __name__ == "__main__":

	main()
