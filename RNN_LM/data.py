import numpy as np
from collections import Counter
import sys

def build_dictionary(file_list, out_path, max_size=0):
    words=[]
    for item in file_list:
        with open(item) as f:
            for line in f:
                words.extend(line.lower().strip().split())
    words=list(zip(*list(Counter(words).most_common())))[0]
    words=['[BOS]','[EOS]','[UNK]']+list(words)
    if max_size<3:
        max_size=len(words)
    with open(out_path,'w') as g:
        g.write('\n'.join(words[:max_size]))

class vocab:
    def __init__(self, vocab_file):
        with open(vocab_file) as f:
            self.words=f.read().split('\n')
            self.word2id=dict(zip(self.words, range(len(self.words))))
            self.id2word=dict(zip(range(len(self.words)), self.words))
            self.vocab_size=len(self.words)
    def encode(self, word_list):
        return [self.word2id[x] if x in self.word2id else self.word2id['[UNK]']  for x in word_list]
    def decode(self, id_list):
        return [self.id2word[x] for x in id_list if x>=2]

class dataset:
    def __init__(self, Vocab, file_path, max_seq_len, max_data_len=0, max_data_word=0):
        self.data=[]
        self.data_len=[]
        with open(file_path) as f:
            counter=0
            s=0
            for line in f:
                if max_data_len>0 and counter>=max_data_len:
                    break
                line=line.lower().strip().split()[:max_seq_len]
                s+=len(line)
                if max_data_word>0 and s>=max_data_word:
                    break
                if len(line)==0:
                    continue
                line=Vocab.encode(line)
                self.data_len.append(min(len(line)+1, max_seq_len))
                for i in range(max_seq_len-len(line)):
                    line.append(Vocab.word2id['[EOS]'])
                self.data.append(line)
                counter+=1
        print(len(self.data))
        self.pointer=0
        self.data=np.array(self.data)
        self.data_len=np.array(self.data_len)
    def get_batch(self, batch_size):
        if self.pointer+batch_size>=len(self.data):
            data_batch=self.data[-batch_size:]
            data_batch_len=self.data_len[-batch_size:]
            self.pointer=0
        else:
            data_batch=self.data[self.pointer:self.pointer+batch_size]
            data_batch_len=self.data_len[self.pointer:self.pointer+batch_size]
            self.pointer+=batch_size
        return data_batch, data_batch_len, self.pointer

if __name__=='__main__':
    out_path=sys.argv[1]
    max_size=int(sys.argv[2])
    file_list=sys.argv[3:]
    build_dictionary(file_list, out_path, max_size)
    
    #Vocab=vocab(out_path)
    #Dataset=dataset(Vocab, file_list[0], 3)
    #for i in range(10):
    #    print(Dataset.get_batch(2))