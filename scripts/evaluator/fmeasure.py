#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\
Created on Tue Dec 14 11:58:52 2021

@author: asep

reference from
[1] https://github.com/nju-websoft/ESBM/tree/master/v1.2
[2] https://github.com/nju-websoft/DeepLENS/blob/master/code/train_test.py
"""
import numpy as np

class FMeasure:
    """The F-Measure is the harmonic mean of the precision and recall"""
    @staticmethod
    def get_score(summ_tids, gold_list):
        """F-score = 2 * (precision * recall) / (precision + recall)"""
        print("gold_list", gold_list)
        print("summ_tids", summ_tids)
        f_list = []
        for gold in gold_list:
            pred_size = len(summ_tids)
            gold_size = len(gold)
            if len(gold) != pred_size:
                print('gold-k:', len(gold), pred_size)

            if pred_size == 0 or gold_size == 0:
                f_list.append(0)
                continue

            corr = len([t for t in summ_tids if t in gold])
            precision = corr / pred_size
            recall = corr / gold_size
            f_score = 2 * ((precision * recall) / (precision + recall)) if corr != 0 else 0
            f_list.append(f_score)
        favg = np.mean(f_list)
        return favg

    @staticmethod
    def get_penalized_score(summ_tids, gold_list, top_k):
        """Completeness-penalized F-score.

        This keeps the base F-measure but downweights it when the model returns
        fewer than top_k triples. Empty summaries therefore score 0.
        """
        if top_k <= 0:
            return 0

        base_score = FMeasure.get_score(summ_tids, gold_list)
        completeness = min(len(summ_tids), top_k) / top_k
        return base_score * completeness
    def __repr__(self):
        return self.__class__.__name__
    