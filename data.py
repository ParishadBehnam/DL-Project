import cv2
import numpy as np
import torch

from config import *
from utils import get_one_hot_from_dict

src_path = ''


class DataLoader:
    def __init__(self):
        max_len = 0
        file = open(src_path + 'Dataset/formulas/train_formulas.txt', 'r')

        # finding vocabs in latex formulas
        vocabs = set()
        for line in file.readlines():
            syms = line.strip().split()
            if len(syms) > max_len:
                max_len = len(syms)
            vocabs.update(syms)

        # vocab indexing
        self.vocab_list = list(vocabs) + ['eof', 'bof']  # end_of_formula and beginning_of_formula
        self.vocab_ids = {x: idx for idx, x in enumerate(self.vocab_list)}
        self.max_len = max_len
        self.vocab_size = len(self.vocab_list)

        self.tensor_target = None
        self.tensor_data = None

    def load_target(self, batch_num):  # target : Latex formula
        formulas = list()
        with open(src_path + 'Dataset/formulas/train_formulas.txt', 'r') as file:
            lines = file.readlines()
            for l in range(batch_num * batch_size, (batch_num + 1) * batch_size):
                line = lines[l]
                syms = line.strip().split()
                formulas.append(syms)

        tensor_target = np.zeros((batch_size, self.max_len + 2, self.vocab_size))
        for i, f in enumerate(formulas):
            tensor_target[i, 0, :] = get_one_hot_from_dict('bof', self.vocab_ids)
            for j in range(len(f)):
                tensor_target[i, j + 1, :] = get_one_hot_from_dict(f[j], self.vocab_ids)
            while j < self.max_len + 1:
                tensor_target[i, j + 1, :] = get_one_hot_from_dict('eof', self.vocab_ids)  # padding formulas with 'eof'
                j += 1
        self.tensor_target = torch.from_numpy(tensor_target).float()

    def load_data(self, batch_num):  # data : Latex formula image
        tensor_data = np.zeros((batch_size, 1, image_h, image_w))
        for i in range(batch_size):
            tensor_data[i, 0, :, :] = cv2.imread(
                '{}Dataset/images/images_train/{}.png'.format(src_path, i + batch_num * batch_size),
                cv2.IMREAD_GRAYSCALE)
        self.tensor_data = torch.from_numpy(tensor_data).float()

    def load_batch(self, batch_num):
        self.load_data(batch_num)
        self.load_target(batch_num)
        return self.tensor_data, self.tensor_target

    @staticmethod
    def load_single_data(id):
        tensor_data = cv2.imread(src_path + 'Dataset/images/images_validation/{}.png'.format(id), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(tensor_data.reshape(1, 1, tensor_data.shape[0], tensor_data.shape[1])).float()
