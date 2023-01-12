# -*- coding:utf-8 -*-

import os
import json
import time
import argparse
import csv
import random
import torch

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


parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='evaluation',
                    help='preprocessing|train|evaluation')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        # self.model_dir = 'saved_models'
        self.model_dir = 'saved_models'

        self.hyper = Hyper(os.path.join('experiments',
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
        if self.exp_name == 'conll_selection_re':
            self.preprocessor = Conll_selection_preprocessing(self.hyper)
        elif self.exp_name == 'data_selection_re':
            self.preprocessor = Conll_selection_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_selection_re':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        elif self.exp_name == 'conll_bert_re':
            self.preprocessor = Conll_bert_preprocessing(self.hyper)
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
            self.evaluation()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        print(self.model_dir,self.exp_name,epoch)
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                             self.exp_name + '_' + str(epoch))))

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
                ner_predict.append(o['decoded_tag'][i])
                ner_gold.append(o['gold_tags'][i])
                triples_predict.append(o['selection_triplets'][i])
                triples_golds.append(o['spo_gold'][i])

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
                if ner_predict[i][j][0] == 'B' and ner_predict[i][j + 1][0] == 'O':
                    w.append(j)
                    word.append(w)
                    w = []
                elif ner_predict[i][j][0] == 'B' and ner_predict[i][j + 1][0] == 'I':
                    w.append(j)
                elif ner_predict[i][j][0] == 'I':
                    w.append(j)
                else:
                    word.append(w)
                    w = []
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


        with open('data/writeDate/triples.csv','w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            tl=[]
            headers=['头实体','关系','尾实体']
            for t in triples_gold:
                for tt in t:
                    ttt=[tt['subject'],tt['predicate'],tt['object']]
                    if ttt not in tl:
                        tl.append(ttt)

            writer.writerow(headers)
            writer.writerows(tl)
            f.close()
        for t in triples_gold:
            for tt in t:
                ttt = [tt['subject'], tt['predicate'], tt['object']]
                if ttt not in tl:
                    tl.append(ttt)

        print("\n语句如下：")
        print(line[19][0][0])
        print("抽取出来的实体关系如下")
        for i in triples_gold[19]:
            print("头实体：" + i['object'] + " 关系：" + i['predicate'] + " 尾实体：" + i['subject'])
       


    def evaluation(self):

        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            out=[]
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])
                out.append(output)
            self.result = out

            self.write_result()
            self.eval_()

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

if __name__ == "__main__":
    config = Runner(exp_name='chinese_selection_re')
    config.run(mode=args.mode)



