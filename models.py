import torch
import torch.nn as nn


class EncoderModel(nn.Module):
    def __init__(self, vocab_size):
        super(EncoderModel, self).__init__()

        self.vocab_size = vocab_size

        # Encoder part som convolution layers and max pooling layers
        encoder_layers = list()

        encoder_layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        encoder_layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1))
        encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        encoder_layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        h = self.encoder(x)
        return torch.mean(h.reshape(h.shape[0], h.shape[1], -1), dim=1)

    def save(self, path='SavedModels/encoderModel'):
        torch.save(self, path)

    @staticmethod
    def load(path='SavedModels/encoderModel'):
        return torch.load(path)


class DecoderModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(DecoderModel, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Decoder part of model, a GRU with a linear for generating symbols of formula
        self.decoder = nn.GRU(self.vocab_size, hidden_size, batch_first=True)
        self.hidden2label = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, hidden):
        lstm_out, new_hidden = self.decoder(x, hidden)
        logit = self.softmax(self.hidden2label(lstm_out))
        return logit, new_hidden

    def save(self, path='SavedModels/decoderModel1'):
        torch.save(self, path)

    @staticmethod
    def load(path='SavedModels/decoderModel1'):
        return torch.load(path)
