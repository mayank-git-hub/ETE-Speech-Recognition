from dataloader import DataLoaderRecog, my_collate
from torch.utils import data
import config
from model import E2E
from tqdm import tqdm
from torch import nn
import torch


def test(model):

	all_loss = []
	all_loss_ctc = []
	all_loss_att = []
	all_cer = []
	all_wer = []

	running_loss = 0
	running_loss_ctc = 0
	running_loss_att = 0
	running_cer = 0
	running_wer = 0

	dataloader = tqdm(data.DataLoader(
		DataLoaderRecog(),
		batch_size=config.test_param['batch_size'],
		num_workers=config.test_param['num_workers'],
		collate_fn=my_collate,
		shuffle=False
	))

	model.eval()

	with torch.no_grad():

		for no, (audio, audio_length, path, text, token, token_id) in enumerate(dataloader):

			if config.use_cuda:
				audio = audio.cuda()
				audio_length = audio_length.cuda()
				token_id = token_id.cuda()

			loss, loss_att, loss_ctc, cer, wer, ys_hat, ys_pad = model(audio, audio_length, token_id)

			all_loss.append(loss.item())
			all_loss_att.append(loss_att.item())
			all_loss_ctc.append(loss_ctc.item())
			all_cer.append(cer)
			all_wer.append(wer)

			running_loss = (running_loss*no + loss.item())/(no + 1)
			running_loss_ctc = (running_loss_ctc*no + loss_ctc.item())/(no + 1)
			running_loss_att = (running_loss_att*no + loss_att.item())/(no + 1)
			running_cer = (running_cer*no + cer)/(no + 1)
			running_wer = (running_wer*no + wer)/(no + 1)

			dataloader.set_description(
				'Avg. Loss: {0:.4f} | '
				'Avg Loss_Att: {1:.4f} | '
				'Avg Loss_CTC: {2:.4f} | '
				'CER: {3:.4f} | '
				'WER: {4:.4f}'.format(
					running_loss,
					running_loss_att,
					running_loss_ctc,
					running_cer,
					running_wer
				)
			)

	return (
		'Avg. Loss: {0:.4f} | '
		'Avg Loss_Att: {1:.4f} | '
		'Avg Loss_CTC: {2:.4f} | '
		'CER: {3:.4f} | '
		'WER: {4:.4f}'.format(
			running_loss,
			running_loss_att,
			running_loss_ctc,
			running_cer,
			running_wer
		))


def get_char_list():

	char_list = ['<blank>']

	with open(config.cache_dir + '/unigram_model/unigram.vocab', 'r') as f:
		for i in f:
			char_list.append(i.split()[0])

	char_list.append('<eos>')

	return char_list


def main():

	args = config.ModelArgs()
	args.report_cer = True
	args.report_wer = True

	char_list = get_char_list()

	model = E2E(idim=80, odim=5002, args=args, char_list=char_list)

	if config.use_cuda:
		model = model.cuda()
		model = nn.DataParallel(model)

	checkpoint = torch.load(config.test_model)
	model.load_state_dict(checkpoint['model'])

	test(model)


if __name__ == "__main__":

	main()
