#!/usr/bin/env python3

# -*- coding:utf-8 -*-

# author  : hzh
# contact : 1006625340@qq.com
# datetime: Created in 2023/1/11 18:51
# software: PyCharm

import os
import json
import time
import argparse
import csv
import random
import torch
import sys
sys.path.append('/workspace/8.2.3-1/1_算法示例')

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD
from pytorch_transformers import AdamW, WarmupLinearSchedule

from lib_8.preprocessings import Chinese_selection_preprocessing, Conll_selection_preprocessing, Conll_bert_preprocessing
from lib_8.dataloaders import Selection_Dataset, Selection_loader
from lib_8.metrics import F1_triplet, F1_ner
from lib_8.models import MultiHeadSelection
from lib_8.config import Hyper


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        # self.model_dir = 'saved_models'
        self.model_dir = 'saved_models'

        self.hyper = Hyper(os.path.join('/workspace/8.2.3-1/1_算法示例/experiments',
                                        self.exp_name + '.json'))

        self.gpu = self.hyper.gpu
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.optimizer = None
        self.model = None
        self.result= None


    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5),
            'adamw': AdamW(model.parameters())
        }
        return m[name]

    def _init_model(self):
        self.model = MultiHeadSelection(self.hyper).cpu()

    def preprocessing(self):
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            res = self.evaluation()
            return res
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        print(self.model_dir,self.exp_name,epoch)
        self.model.load_state_dict(
            torch.load(
                os.path.join('/workspace/8.2.3-1/1_算法示例/saved_models/chinese_selection_re_9')))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, self.exp_name + '_' + str(epoch)))

    def eval_(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        out = self.result
        ner_predict = []
        ner_gold=[]
        triples_predict = []
        triples_golds = []
        triples_gold = []


        for o in out:
            for i in range(len(o['decoded_tag'])):
                raise NotImplementedError('请补全代码块3，详情见注释')
                # 将 o 中 'decoded_tag' 列表中索引为 i 的元素添加到 ner_predict 列表中。
                # 将 o 中 'gold_tags' 列表中索引为 i 的元素添加到 ner_gold 列表中。
                # 将 o 中 'selection_triplets' 列表中索引为 i 的元素添加到 triples_predict 列表中。
                # 

        for ts in triples_golds:
            ttt = []
            if len(ts) != 0:
                for t in ts:
                    tt = {}
                    subject = ''.join(str(s) for s in t['subject'])
                    object = ''.join(str(s) for s in t['object'])
                    tt['object'] = object
                    tt['predicate'] = t['predicate']
                    tt['subject'] = subject
                    ttt.append(tt)
            else:
                ttt = []
            triples_gold.append(ttt)


        tri_all=0
        tri_get=0
        ner_count = 0

        ner_hit = 0
        tri_hit = 0

        all_type = []
        tri=[]
        tri_=[]

        for t in triples_gold:
            for tt in t:
                tri.append(tt)
                tri_all=tri_all+1
        for t_ in triples_predict:
            for tt_ in t_:
                tri_.append(tt_)
                tri_get=tri_get+1

        for t in tri_:
            if t in tri:
                tri_hit=tri_hit+1


        for i in range(len(dev_set)):
            sentence = ''.join(str(s) for s in dev_set.text_list[i])

            type=[]
            for t in triples_predict[i]:
                s_ind=sentence.find(t['subject'])
                o_ind = sentence.find(t['object'])
                s_type=ner_predict[i][s_ind][2:]
                o_type=ner_predict[i][o_ind][2:]

                type.append([o_type,s_type])
            all_type.append(type)



            for j in range(len(ner_gold[i])):
                ner_count=ner_count+1
                if ner_predict[i][j]==ner_gold[i][j]:
                    ner_hit=ner_hit+1


    def write_result(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        out=self.result

        ner_predict = []
        ner_gold = []
        triples_predict = []
        triples_golds = []
        # 存储了所有三元组，并且按序号排列了
        triples_gold = []


        c = 0
        for o in out:
            for i in range(len(o['decoded_tag'])):
                ner_predict.append(o['decoded_tag'][i])
                ner_gold.append(o['gold_tags'][i])
                triples_predict.append(o['selection_triplets'][i])
                triples_golds.append(o['spo_gold'][i])
                c = c + 1


        for ts in triples_golds:
            ttt = []
            if len(ts) != 0:
                for t in ts:
                    tt = {}
                    subject = ''.join(str(s) for s in t['subject'])
                    # print(subject)
                    object = ''.join(str(s) for s in t['object'])
                    # print(object)
                    tt['object'] = object
                    tt['predicate'] = t['predicate']
                    tt['subject'] = subject
                    ttt.append(tt)
            else:
                ttt = []
            triples_gold.append(ttt)

        line=[]
        all_type=[]
        for i in range(len(dev_set)):
            sentence = ''.join(str(s) for s in dev_set.text_list[i])

            type = []
            for t in triples_predict[i]:
                s_ind = sentence.find(t['subject'])
                o_ind = sentence.find(t['object'])
                s_type = ner_predict[i][s_ind][2:]
                o_type = ner_predict[i][o_ind][2:]
                type.append([o_type, s_type])
            all_type.append(type)

            word = []
            w = []
            for j in range(len(ner_predict[i])):
                raise NotImplementedError('请补全代码块2，详情见注释')
                # 检查当前字符串的第一个字符是否为 'B' 和下一个字符是否为 'O'。
                # 如果是这种情况，将当前索引 j 添加到 w 列表中，并将 w 列表添加到 word 列表中，然后将 w 列表重置为空。
                # 如果当前字符串的第一个字符为 'B' 且下一个字符为 'I'，它将当前索引 j 添加到 w 列表中。
                # 如果当前字符串的第一个字符为 'I'，它将当前索引 j 添加到 w 列表中
                # 如果没有以上情况，那么它将 w 列表添加到 word 列表中，并将 w 列表重置为空
                
            words = []
            for w in word:
                if len(w) != 0:
                    words.append(w)
            List = []

            words_List = []
            if len(words) != 0:
                for w in words:
                    words_L = []
                    for ww in w:
                        words_L.append(dev_set.text_list[i][ww])
                    words = ''.join(str(s) for s in words_L)
                    words_List.append(words)

            List.append([sentence, words_List, triples_predict[i],all_type[i]])

            line.append(List)

        return triples_gold[0][0]['object']


    def evaluation(self):

        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            out=[]
            for batch_ndx, sample in pbar:
                raise NotImplementedError('请补全代码块1，详情见注释')
                # 使用 self.model 对象处理输入 sample，将输出赋值给变量 output。
                # 使用 self.triplet_metrics 函数对 output 中的 'selection_triplets' 和 'spo_gold' 进行评估。
                # 使用 self.ner_metrics 函数对 output 中的 'gold_tags' 和 'decoded_tag' 进行评估。
                # 将 output 添加到 out 列表中
            self.result = out

            res = self.write_result()
            self.eval_()

        return res

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=True)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:
                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()
                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))
            self.save_model(epoch)

            if epoch % self.hyper.print_epoch == 0 and epoch > 3:
                self.evaluation()

def solution():
    config = Runner(exp_name="chinese_selection_re")
    res = config.run(mode="evaluation")
    return res

