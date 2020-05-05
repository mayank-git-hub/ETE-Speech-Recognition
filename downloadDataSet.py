import config
import os


def download_dataset():

	print('Stage -1')

	for list_i in config.list_to_download:

		if not os.path.exists(config.path_to_download + '/' + list_i):

			if not os.path.exists(config.path_to_download + '/' + list_i + '.tar.gz'):
				os.system("axel -a -n -10 -o "+config.path_to_download+'/'+list_i+'.tar.gz '+config.url_base+'/'+list_i+'.tar.gz')

			os.system('tar -xvzf ' + config.path_to_download + '/' + list_i + '.tar.gz -C ' + config.path_to_download)

			os.remove(config.path_to_download + '/' + list_i + '.tar.gz')


if __name__ == "__main__":

	download_dataset()
