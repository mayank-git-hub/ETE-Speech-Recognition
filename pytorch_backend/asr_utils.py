#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import logging
# matplotlib related
import os

# io related
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')


# * -------------------- general -------------------- *
class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def __getitem__(self, name):
        return self.obj[name]

    def __len__(self):
        return len(self.obj)

    def fields(self):
        return self.obj

    def items(self):
        return self.obj.items()

    def keys(self):
        return self.obj.keys()


def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json)

    :param str model_path: model path
    :param str conf_path: optional model config path
    """

    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        return json.load(f, object_hook=AttributeDict)


def torch_save(path, model):
    """Function to save torch model states

    :param str path: file path to be saved
    :param torch.nn.Module model: torch model
    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    """Function to load torch model states

    :param str path: model file or snapshot file to be loaded
    :param torch.nn.Module model: torch model
    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text string
    :return: recognition token string
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        if len(js['output']) > 0:
            out_dic = dict(js['output'][0].items())
        else:
            # for no reference case (e.g., speech translation)
            out_dic = {'name': ''}

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            if 'text' in out_dic.keys():
                logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])

    return new_js


def plot_spectrogram(plt, spec, mode='db', fs=None, frame_shift=None,
                     bottom=True, left=True, right=True, top=False,
                     labelbottom=True, labelleft=True, labelright=True,
                     labeltop=False, cmap='inferno'):
    """Plot spectrogram using matplotlib

    :param matplotlib.pyplot plt:
    :param np.ndarray spec: Input stft (Freq, Time)
    :param str mode: db or linear.
    :param int fs: Sample frequency. To convert y-axis to kHz unit.
    :param int frame_shift: The frame shift of stft. To convert x-axis to second unit.
    :param bool bottom:
    :param bool left:
    :param bool right:
    :param bool top:
    :param bool labelbottom:
    :param bool labelleft:
    :param bool labelright:
    :param bool labeltop:
    :param str cmap: colormap defined in matplotlib

    """
    spec = np.abs(spec)
    if mode == 'db':
        x = 20 * np.log10(spec + np.finfo(spec.dtype).eps)
    elif mode == 'linear':
        x = spec
    else:
        raise ValueError(mode)

    if fs is not None:
        ytop = fs / 2000
        ylabel = 'kHz'
    else:
        ytop = x.shape[0]
        ylabel = 'bin'

    if frame_shift is not None and fs is not None:
        xtop = x.shape[1] * frame_shift / fs
        xlabel = 's'
    else:
        xtop = x.shape[1]
        xlabel = 'frame'

    extent = (0, xtop, 0, ytop)
    plt.imshow(x[::-1], cmap=cmap, extent=extent)

    if labelbottom:
        plt.xlabel('time [{}]'.format(xlabel))
    if labelleft:
        plt.ylabel('freq [{}]'.format(ylabel))
    plt.colorbar().set_label('{}'.format(mode))

    plt.tick_params(bottom=bottom, left=left, right=right, top=top,
                    labelbottom=labelbottom, labelleft=labelleft,
                    labelright=labelright, labeltop=labeltop)
    plt.axis('auto')
