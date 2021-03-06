# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import random


def get_sequence_label(start_id , end_id , sequence_len):
    '''
    start_id 和end_id 是0base的
    '''
    front = np.zeros(start_id)
    #print("start_id = ",start_id)
    #print("type = ",type(start_id))
    #print("end_id = ",end_id) 
    #print("end_id - start_id = ",end_id - start_id)
    middle = np.ones(end_id - start_id + 1)
    back = np.zeros(sequence_len - end_id - 1)
    res = np.concatenate([front,middle,back],axis = -1)
    #res.dtype = np.int32
    #print("1 =",res[start_id])
    #print("2 = ",res[end_id])
    #print("11 = ",res[start_id:])
    #print("12 = ",res[start_id:end_id + 1])
    #print("22 = " ,res[:end_id+1])
    #print("01 = ",res[:start_id])
    #print("20 = ",res[end_id + 1:])
    #print(len(res))
    #print(sequence_len)
    #exit(89)
    assert len(res) == sequence_len
    return res

class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,max_word_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_word_len = max_word_len
        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0]
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                data_set.append(sample)
        return data_set






    def _one_mini_batch(self, data, indices, pad_id , pad_char_id = None):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],

                      'question_char_ids':[],
                      'question_char_length':[],
                      'passage_char_ids':[],
                      'passage_char_length':[],

                      'start_id': [],
                      'end_id': [],
                      'real_pass':[],
                      'sequence_label':[]}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)

        #passages 和对应的 start_id end_id 是分别append的，所以如果要start_id 和end_id的位置随机，就需要将这两个同时加入
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    if pad_char_id is not None:
                        batch_data["question_char_ids"].append(sample['question_char_ids'])
                        # sample_question_length = [min(len(sa_qu) , self.max_word_len)for sa_qu in sample['question_char_ids']]
                        # batch_data["question_length"].append(sample_question_length)

                        sample_passage_char_ids = sample['passages'][pidx]['passage_char_ids']
                        batch_data['passage_char_ids'].append(sample_passage_char_ids)
                        # sample_passage_length = [min(len(sa_pa), self.max_word_len) for sa_pa in sample_passage_char_ids]
                        # batch_data['passage_char_length'].append(sample_passage_length)


                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    if pad_char_id is not None:
                        batch_data['question_char_ids'].append([[]])
                        batch_data['question_char_length'].append([0])
                        batch_data['passage_char_ids'].append([[]])
                        batch_data['passage_char_length'].append([0])

        # 一个start end index对，对应max_passage_num 个 passage，所以一旦要random就可以这么办
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, pad_char_id = pad_char_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                start_id = min(padded_p_len - 1 , sample['answer_spans'][0][0])
                end_id = min(padded_p_len - 1 , sample['answer_spans'][0][1])
                batch_data['start_id'].append(gold_passage_offset + start_id)
                batch_data['end_id'].append(gold_passage_offset + end_id)
                batch_data["real_pass"].append(sample['answer_passages'][0])
                batch_data['sequence_label'].append(get_sequence_label(start_id = batch_data['start_id'][-1] , end_id = batch_data['end_id'][-1], sequence_len = max_passage_num * padded_p_len))

            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
                batch_data['real_pass'].append(0)
                batch_data['sequence_label'].append(np.zeros(padded_p_len * max_passage_num))

        return batch_data

    def _zq_one_mini_batch(self, data, indices, pad_id,pad_char_id = None):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': [],
                      'real_pass':[],
                      'sequence_label':[]}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        for idx, sample in enumerate( batch_data['raw_data'] ):
            if 'answer_passages' in sample and len(sample['answer_passages']):
                start_num = idx * max_passage_num#举个例子比如说21～25
                end_num = start_num + max_passage_num
                random_indices = list(range(start_num , end_num))
                random.shuffle(random_indices)
                temp_question,temp_passage = [] , []
                temp_q_len , temp_p_len = [] , []

                answer_doc_after_shuffle = -1
                for real_idx, i in enumerate( random_indices ):
                    temp_question.append(batch_data["question_token_ids"][i])
                    temp_passage.append(batch_data["passage_token_ids"][i])
                    temp_p_len.append(batch_data['passage_length'][i])
                    temp_q_len.append(batch_data['question_length'][i])
                    if i - start_num == sample['answer_passages'][0]:#sample['answer_passages'][0]中装的是正确的答案的位置
                        answer_doc_after_shuffle = real_idx #random.shuffle()后原来的数字num（num=3，for example），可能被shuffle到别的位置去了，用real_idx 记录一下位置

                if answer_doc_after_shuffle == -1 :
                    print( "Error asnwer_doc_after_shuffle should not be -1")
                    print("sample['answer_passages'][0] = ",sample['answer_passages'][0])
                    print("question_id = ",sample["question_id"])
                    exit(1778)

                batch_data["question_token_ids"][start_num:end_num] = temp_question
                batch_data["passage_token_ids"][start_num:end_num] = temp_passage
                batch_data["passage_length"][start_num:end_num] = temp_p_len
                batch_data["question_length"][start_num:end_num] = temp_q_len

                gold_passage_offset = padded_p_len * answer_doc_after_shuffle
                start_id = min(padded_p_len - 1 , sample['answer_spans'][0][0])
                end_id = min(padded_p_len - 1 , sample['answer_spans'][0][1])
                batch_data['start_id'].append(gold_passage_offset + start_id)
                batch_data['end_id'].append(gold_passage_offset + end_id)
                batch_data['real_pass'].append(answer_doc_after_shuffle)
                #start_id = batch_data['start_id'][-1]  
                #end_id = batch_data['end_id'][-1]
                #sequence_len = max_passage_num * padded_p_len
                
                batch_data['sequence_label'].append(get_sequence_label(start_id = batch_data['start_id'][-1] , end_id = batch_data['end_id'][-1], sequence_len = max_passage_num * padded_p_len))
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
                batch_data['real_pass'].append(0)
                batch_data['sequence_label'].append(np.zeros(padded_p_len * max_passage_num))
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id , pad_char_id = None):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length'])) #这个batch中最长的p的长度
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        if pad_char_id != None:
            '''
            >>> a = [['Hello',"My","name"],['is','Zhengqun'],[]]
            >>> b = [map(lambda word:len(word),sent)for sent in a]
            >>> b
            [[5, 2, 4], [2, 8], []]
            >>> aa = [[['H','e','l','l','0'],['M','y'],['n','a','m','e']],[['i','s'],['Z','h','e','n','g']],[]]
            >>> b = [map(lambda word:len(word),sent)for sent in aa]
            >>> b
            [[5, 2, 4], [2, 5], []]
            >>> b = [map(lambda word:min(len(word) , 4),sent)for sent in aa]
            >>> b
            [[4, 2, 4], [2, 4], []]
            '''
            batch_data['passage_char_length'] = [map(lambda word : min(len(word), self.max_word_len) , sent) for sent in batch_data['passage_char_ids'] ]
            batch_data['question_char_length'] = [map(lambda word : min(len(word) , self.max_word_len) , sent) for sent in batch_data['question_char_ids']]
            #这里的word其实是char_list_of_a_word
            pad_p_word_len = min(self.max_word_len, max(map(max , batch_data['passage_char_length'])))
            pad_q_word_len = min(self.max_word_len, max(map(max , batch_data['question_char_length'])))


            passage_char_ids = np.zeros([len(batch_data['passage_char_ids']) , pad_p_len , pad_p_word_len],dtype=np.int32)
            passage_char_len = np.zeros([len(batch_data['passage_char_ids']) , pad_p_len], dtype = np.int32)

            for i , line in enumerate(batch_data['passage_char_ids']):
                for j , word in enumerate(line):
                    if j >= pad_p_len:
                        break#应该永远不会执行
                    passage_char_len[i,j] = min(self.max_word_len , len(word))
                    for k , char in enumerate(word):
                        passage_char_ids[i,j,k] = char
            batch_data['passage_char_ids'] = passage_char_ids
            batch_data['passage_char_length'] = passage_char_len

            question_char_ids = np.zeros([len(batch_data['question_char_ids']) , pad_q_len , pad_q_word_len],dtype = np.int32)
            question_char_len = np.zeros([len(batch_data['question_char_ids']) , pad_q_len], dtype=np.int32)
            for i , line in enumerate(batch_data["question_char_ids"]):
                for j , word in enumerate(line):
                    if j >= pad_q_len:
                        break
                    question_char_len[i,j] = min(self.max_word_len , len(word))
                    for k , char in enumerate(word):
                        question_char_ids[i,j,k] = char
            batch_data['question_char_ids'] = question_char_ids
            batch_data["question_char_length"] = question_char_len
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def char_iter(self , set_name = None):
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    for char in token :
                        yield char
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        for char in token :
                            yield char

    def convert_to_ids(self, vocab,use_char_level = False):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'], sample['question_char_ids'] = vocab.convert_to_ids(sample['question_tokens'] , use_char_level = use_char_level)
                #前者是一维的【1，2，3，5】，后者是二维的，每个单词中的每个字都单独成为一个元素
                for passage in sample['passages']:
                    passage['passage_token_ids'], passage['passage_char_ids'] = vocab.convert_to_ids(passage['passage_tokens'] , use_char_level = use_char_level)


    def gen_mini_batches(self, set_name, batch_size, pad_id,pad_char_id=None, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id,pad_char_id)
            #yield self._zq_one_mini_batch(data, batch_indices, pad_id,pad_char_id)

