import numpy as np


def get_one_hot_from_dict(sym, vocab_ids):
    one_hot = np.zeros_like(vocab_ids.keys())
    one_hot[vocab_ids[sym]] = 1
    return one_hot
