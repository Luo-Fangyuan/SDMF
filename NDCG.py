# import torch
import math
import functools
import numpy as np

@functools.lru_cache(100)
def IDCG_at(K: int, label, implict=False):
    if len(label) > K:
        K = K
    else:
        K = len(label)
    if implict == True:
        return sum([1 / math.log2(k + 2) for k in range(K)])
    else:
        sort_label, sort_ind = torch.sort(label, descending=True)
        return sum([(2 ** sort_label[k] - 1) / math.log2(k + 2) for k in range(K)])

def DCG_at(label, pred, K: int, implict = False):
    reclist = torch.argsort(pred, descending=True)
    if implict == True:
        return sum([1 / math.log2(k + 2) for k in range(K) if 0 < label[reclist[k]]])
    else:
        return sum([(2 ** label[reclist[k]] - 1) / math.log2(k + 2) for k in range(K)])

def NDCG_at(label, pred, K: int):
    if len(label) > K:
        return DCG_at(label, pred, K) / IDCG_at(K, label, implict=False)
    else:
        K = len(label)
        return DCG_at(label, pred, K) / IDCG_at(K, label, implict=False)


def weight_NDCG(K, k, data):
    return 1.0 / (math.log2(k + 2) * IDCG_at(K, data, implict=False))

def calDCG_k(dictdata, k):
    nDCG = []
    for key in dictdata.keys():
        listdata = dictdata[key]
        real_value_list = sorted(listdata, key=lambda x: x[1], reverse=True)
        idcg = 0.0
        predict_value_list = sorted(listdata, key=lambda x: x[0], reverse=True)
        dcg = 0.0
        if len(listdata) >= k:
            for i in range(k):
                idcg += (pow(2, real_value_list[i][1]) - 1) / (math.log2(i + 2))
                dcg += (pow(2, predict_value_list[i][1]) - 1) / (math.log2(i + 2))
            if (idcg != 0):
                nDCG.append(float(dcg / idcg))
            else:
                nDCG.append('Null!')
        else:
            continue
    ave_ndcg = '%.5f'%(np.mean(nDCG))
    return ave_ndcg

