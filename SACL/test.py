import OpenAttack
from OpenAttack.utils import Dataset, DataInstance
import random
import codecs
import json
import numpy as np
import pickle
import torch
import os

# dataset = OpenAttack.DataManager.load("Dataset.SST.sample")
# print(dataset)
# print(dataset.data())
# print(dataset.data()[1])

# for data_name in OpenAttack.DataManager.AVAILABLE_DATAS:
#     print(data_name)

# shuzu=np.array([True,True,False,False,False,])
# shuzu1=np.array([True,False,False,False,False,])
# semantic_sims=np.array([0.5,0.4,0.3,0.2,0.05])
# shuzu *= (semantic_sims >= 0.1)
# print(shuzu*shuzu1)
#
# print(np.sum(shuzu))

# for i in data_list:
#     print(i)

# Test OpenAttack.DataManager
# train, valid, test = OpenAttack.loadDataset("SST")
# print(train)
# print(type(train))
# print(train[0])


# test Dataset
# def mapping(data):
#     return Dataset([
#         DataInstance(
#             x=it[0],
#             y=it[1]
#         ) for it in data
#     ], copy=False)
# train, valid, test = pickle.load(open('/home/lwd/cstools/project/TextAattck/data/Dataset.SST', "rb"))
# print(train)
# # print(mapping(train))
# databyDataset=mapping(train)
# print(databyDataset)
# # for i in databyDataset:
# #     print(i)

# test pytorch slice
# test_tensor=torch.Tensor([[1,2],
#                           [3,4],
#                           [5,6]])
# test_tensor_1=torch.Tensor([1,2])
# test_tensor_2=torch.Tensor([3,4])
# print(test_tensor[0:2])
# print(len(test_tensor))
# print(test_tensor.t().shape)
# print(torch.mul(test_tensor_1.t(),test_tensor_2))

# tensor1=torch.tensor(1.0351)
# tensor2=torch.tensor(0.548)
# print((tensor1/float(tensor2)).log())
# print(-tensor1)

# tensor1=torch.tensor([1.0,2.0])
# def norm_vector(embedding, p=2):
#     norm = torch.norm(embedding, p=2)
#     return embedding / norm
# print(norm_vector(tensor1))

# tensor1=torch.tensor([1.0,2.0,3,4])
# print(torch.split(tensor1,2)[0])
# print(type(torch.split(tensor1,2)))

# str1='xxx'
# print(isinstance(str1,str))

def prepare_data():
    vocab = {
        "<UNK>": 0,
        "<PAD>": 1
    }
    train, valid, test = OpenAttack.loadDataset("SST")
    # print(train)
    tp = OpenAttack.text_processors.DefaultTextProcessor()
    for dataset in [train, valid, test]:
        for inst in dataset:
            inst.tokens = list(map(lambda x:x[0], tp.get_tokens(inst.x)))
            for token in inst.tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return train, valid, test, vocab
train, valid, test, vocab = prepare_data()
min=10
for data in test:
    if len(data.data()['meta']['tokens'])<4:
        test.remove(data.data()['idx'])
for data in test:
    if len(data.data()['meta']['tokens'])<min:
        min=len(data.data()['meta']['tokens'])

print(min)