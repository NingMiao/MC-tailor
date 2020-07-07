import glob
import numpy as np
import os
import tensorflow as tf
import tqdm



def load_dataset(enc, path, seq_len):
    start_id=enc.encoder['<|endoftext|>']
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    data_list = []
    for path in tqdm.tqdm(paths):
        # Plain text
        with open(path) as f:
            for line in f:
                if len(line.strip())>0:
                    data_list.append([start_id]+enc.encode(line.strip())+[start_id])
    data_len=[min(len(item), seq_len) for item in data_list]
    for i in range(len(data_list)):
        if len(data_list[i])>=seq_len:
            data_list[i]=data_list[i][:seq_len]
        else:
            for j in range(seq_len-len(data_list[i])):
                data_list[i].append(start_id)
    return data_list, data_len


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, data_list, data_len, seed=None):
        self.data_list=np.array(data_list)
        self.data_len=np.array(data_len)
        self.rs = np.random.RandomState(seed=seed)
        self.total_size=len(self.data_list)

    def sample(self, batch_size):
        ind=self.rs.randint(0, len(self.data_list), [batch_size])
        return self.data_list[ind], self.data_len[ind]
