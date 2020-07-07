import math
import sys
import fractions
import warnings
from collections import Counter

from nltk.util import ngrams

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

import json
from tqdm import tqdm
import numpy as np
import pickle as pkl
import gzip as gz


def my_bleu(
        references,
        hypotheses,
        weights=(0.5,0.5),
        smoothing_function=None,
):
    part_id=0
    reference_counts = [[] for _ in range(len(weights))]
    for n in range(1, len(weights)+1):
        for reference in references:
            reference_counts[n-1].append((
                Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
            ))
    ref_lens = [len(reference) for reference in references]
    s_list=[]
    for hypothesis in tqdm(hypotheses, desc=f'worker-{part_id}'):
        p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
        p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.

        for i, _ in enumerate(weights, start=1):
            p_i = my_precision(reference_counts[i-1], hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        hyp_lengths = len(hypothesis)
        ref_lengths = my_closest_ref_length(ref_lens, hyp_lengths)

        bp = brevity_penalty(ref_lengths, hyp_lengths)

        p_n = [
            Fraction(p_numerators[i], p_denominators[i], _normalize=False)
            for i, _ in enumerate(weights, start=1)
        ]

        if p_numerators[1] == 0:
            s_list.append(0)
        else:
            if not smoothing_function:
                smoothing_function = SmoothingFunction().method0

            p_n = smoothing_function(
                p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
            )
            s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
            s = bp * math.exp(math.fsum(s))
            s_list.append(s)
    return np.mean(s_list)


def my_precision(reference_counts, hypothesis, n):
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    max_counts = {}
    for ref_counts in reference_counts:
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), ref_counts[ngram])
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }
    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def my_closest_ref_length(ref_lens, hyp_len):
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
    )
    return closest_ref_len


def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)


class SmoothingFunction:
    def __init__(self, epsilon=0.1, alpha=5, k=5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """
        No smoothing.
        """
        p_n_new = []
        for i, p_i in enumerate(p_n):
            if p_i.numerator != 0:
                p_n_new.append(p_i)
            else:
                _msg = str(
                    "\nThe hypothesis contains 0 counts of {}-gram overlaps.\n"
                    "Therefore the BLEU score evaluates to 0, independently of\n"
                    "how many N-gram overlaps of lower order it contains.\n"
                    "Consider using lower n-gram order or use "
                    "SmoothingFunction()"
                ).format(i + 1)
                warnings.warn(_msg)
                # When numerator==0 where denonminator==0 or !=0, the result
                # for the precision score should be equal to 0 or undefined.
                # Due to BLEU geometric mean computation in logarithm space,
                # we we need to take the return sys.float_info.min such that
                # math.log(sys.float_info.min) returns a 0 precision score.
                p_n_new.append(sys.float_info.min)
        return p_n_new



if __name__ == '__main__':
    f1='/mnt/cephfs_hl/common/lab/miaoning/nmt/workspace1/nmt_data/tst2012.en'
    f2='/mnt/cephfs_hl/common/lab/miaoning/nmt/workspace1/nmt_data/tst2013.en'
    with open(f1) as f:
        data1=[]
        for line in f:
            data1.append(line.strip().split())
    with open(f2) as f:
        data2=[]
        for line in f:
            data2.append(line.strip().split())
    print(my_bleu(data1[:100], data1[0:100]))