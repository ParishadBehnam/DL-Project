import numpy as np
import torch
import torch.nn as nn

from config import *
from data import DataLoader
from models import EncoderModel, DecoderModel
from utils import get_one_hot_from_dict


def train(models, loader):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # parameters : list of parameters of both Encoder and Decoder models
    optimizer = torch.optim.Adam(list(models[0].parameters()) + list(models[1].parameters()), lr=learning_rate)

    models[0].train()  # Encoder
    models[1].train()  # Decoder
    for iter in range(iterations):
        for batch_id in range(train_sample // batch_size):
            data_batch, target_batch = loader.load_batch(batch_id)

            optimizer.zero_grad()
            c = models[0](data_batch)  # the hidden state
            logits, _ = models[1](target_batch, c.reshape(1, c.shape[0], c.shape[1]))
            target_indices = torch.max(target_batch, 2)[1]
            logits = logits.permute(0, 2, 1)
            loss = criterion(logits[:, :, :-1], target_indices[:, 1:])  # cross entropy loss
            print('iter', iter, 'batch', batch_id, 'loss', loss.item())

            if batch_id % 100 == 0:
                with open('save.txt', 'r') as file:
                    flag = file.readlines()[0]
                    if flag.startswith('1'):
                        models[0].save(path='SavedModels/encoderModel' + str(batch_id))
                        models[1].save(path='SavedModels/decoderModel' + str(batch_id))
                        print('models saved')
            loss.backward()
            nn.utils.clip_grad_norm_(list(models[0].parameters()) + list(models[1].parameters()), 10)
            optimizer.step()


def predict(models, vocab_list, vocab_ids, pic):
    models[0].train(False)
    models[1].train(False)
    c = models[0](pic)  # hidden state of pic
    c = c.reshape(1, c.shape[0], c.shape[1])
    words = [torch.from_numpy(get_one_hot_from_dict('bof', vocab_ids)).float()]  # at first, we have only 'bof' in the sentence
    counter = 0
    while vocab_list[torch.argmax(words[-1])] != 'eof' and counter < 100:
        counter += 1
        logits, new_c = models[1](words[-1].reshape(1, 1, words[-1].shape[0]), c)
        logits = torch.exp(logits)  # because of LogSoftmax
        choice = np.random.choice(range(len(vocab_list)), p=logits[0, 0, :].detach().numpy())  # sample from the logits
        words.append(torch.from_numpy(get_one_hot_from_dict(vocab_list[choice], vocab_ids)).float())
        c = new_c  # updating hidden state

    formula = ''
    for w in words:
        formula += (' ' + vocab_list[torch.argmax(w)])
    return formula


if __name__ == '__main__':
    loader = DataLoader()
    if TRAIN:
        enc_model = EncoderModel(loader.vocab_size)
        dec_model = DecoderModel(loader.vocab_size, hidden_size)
        # enc_model = EncoderModel.load()   # while a saved model exists
        # dec_model = DecoderModel.load()
        train([enc_model, dec_model], loader)
        enc_model.save()
        dec_model.save()
    else:
        enc_model = EncoderModel.load(path='SavedModels/encoderModel0')
        dec_model = DecoderModel.load(path='SavedModels/decoderModel0')
        with open("validation_predicted_formulas.txt", "a") as myfile:
            for i in range(validation_sample):
                print('predicting data', i)
                f = predict([enc_model, dec_model], loader.vocab_list, loader.vocab_ids,
                            loader.load_single_data(i))[4:-4] + '\n'  # ignoring 'bof ' and ' eof'
                myfile.write(f)
