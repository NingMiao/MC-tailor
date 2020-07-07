import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from time import time
import bleu

def calculate_bleu_simple(refs, hyp):
    old=[1],[0,1],[0,0,1],[0,0,0,1],
    weights=[[0.25,0.25,0.25,0.25]]
    result=[]
    for weight in weights:
      result.append(sentence_bleu(refs, hyp , weights=weight, smoothing_function=SmoothingFunction().method1))
    return result
    
class corpus_BLEU_old:
    def __init__(self, path):
        self.ref_list=[]
        with open(path) as f:
            for line in f:
                self.ref_list.append(line.strip().split())
    def __call__(self, hyp_list):
        result=[]
        for item in hyp_list:
            item=item.strip().split()
            result.append(calculate_bleu_simple(self.ref_list, item))
        result=list(zip(*result))
        result=[np.mean(x) for x in result]
        return result

class corpus_BLEU:
    def __init__(self, path):
        self.ref_list=[]
        with open(path) as f:
            for line in f:
                self.ref_list.append(line.strip().split())
    def __call__(self, hyp_list):
        result=[]
        hyp_list=[x.strip().split() for x in hyp_list]
        return bleu.my_bleu(hyp_list, self.ref_list)

if __name__=='__main__':
    BLEU_cal=corpus_BLEU('/mnt/cephfs_hl/common/lab/miaoning/nmt/workspace1/nmt_data/tst2012.en')
    with open('/mnt/cephfs_hl/common/lab/miaoning/nmt/workspace1/nmt_data/tst2013.en') as f:
        t=time()
        for i in range(100):
            BLEU_cal(f.readline())
            print(i)
        print(time()-t)
    