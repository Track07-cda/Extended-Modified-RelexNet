import numpy as np
import pandas as pd
import torch
from preprocessor import Preprocessor
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as Data
from modifiers import Modifiers
from graphHandler import GraphHandler

class DataLoader:
    def __init__(self, data, config, graph_handler, preprocessor=Preprocessor(), enc_tokens = OneHotEncoder(), enc_modifiers = OneHotEncoder(), modifier = Modifiers(), dev = 'cpu'):
        self.df = data
        self.dev = dev
        #---------------
        self.batch_size = config['model']['batchSize']
        self.context_window = config['model']['contextWindowSize']
        #---------------
        self.modifiers = modifier.BOOSTER_DICT.keys()
        self.enc_modifiers = enc_modifiers
        self.enc_tokens = enc_tokens
        # self.vocab_frequency = 5
        #---------------
        self.padding_length = config['model']['paddingLength']
        self.fold_num = config['model']['foldNum']
        self.graph_handler = graph_handler
        #---------------
        # self.first_sentence = 128
        # self.second_sentence = self.padding_length - self.first_sentence
        self.count = 0


        self.apply_preprocessor(preprocessor)
        self.enc_tokens.fit(self.vocab)
        self.enc_modifiers = enc_modifiers
        self.enc_modifiers.fit(self.modifier_id_to_fit)
        #---------------
        self.split(1 - config['model']['testPortion'], config['model']['testPortion'])
        #---------------
        train_num = len(self.train)
        if train_num % self.batch_size == 0:
            self.no_batch = train_num / self.batch_size
        else:
            self.no_batch = int(train_num / self.batch_size) + 1

        if 'train' in config['mode']:
            trainX, trainY, train_meta, train_meta_wi = self.build_training_data(enc_tokens, self.train, is_exp=False)
            testX, testY, test_meta, test_meta_wi = self.build_training_data(enc_tokens, self.test, is_exp=False)
            print(f'Train Dataset Shape : {trainX.shape}')
            print(f'Test Dataset Shape : {testX.shape}')
            print(f'Next Modifier : {self.count}')
            self.train_dataset = self.get_batch(trainX, trainY, train_meta, train_meta_wi)
            self.test_dataset = self.get_batch(testX, testY, test_meta, test_meta_wi)
        if 'test' in config['mode']:
            self.eval = self.df.iloc[config['evaluationRange']['begin']:config['evaluationRange']['end']]
            testX, testY, test_meta, test_meta_wi = self.build_training_data(self.enc_tokens, self.eval, is_exp=True)
            print(f'Test Dataset Shape : {testX.shape}')
            self.test_dataset = self.get_batch(testX, testY, test_meta, test_meta_wi)


    def get_batch(self, X, Y, meta, meta_wi):
        dataset = Data.TensorDataset(X, Y, meta, meta_wi)
        loader = Data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=2,
        )
        return loader


    def remove_low_frequency(self, list):
        new_list = []
        for x in list:
            if x in self.token_to_count.keys():
                new_list.append(x)
        return new_list


    def apply_preprocessor(self, preprocessor):
        self.df['tokens'] = [preprocessor(s) for s in self.df['sentence']]
        self.df['tokens'] = [x[:self.padding_length]  if len(x) > self.padding_length else x for x in self.df['tokens']]

        # for index, row in self.df.iterrows():
        #     if len(row['tokens']) > self.max_length:
        #         row[toke]

        ######## Changed part for negatives ###########
        self.token_to_count = Counter([x for l in self.df['tokens'] for x in l if x not in self.modifiers])
        self.modifiers_to_count = Counter([x for l in self.df['tokens'] for x in l if x in self.modifiers])
        ##############      End       #################

        # tmp_token_to_count = self.token_to_count.copy()
        # for index, value in tmp_token_to_count.items():
        #     if value <= self.vocab_frequency:
        #         self.token_to_count.pop(index)
        # self.df['tokens'] = [self.remove_low_frequency(x) for x in self.df['tokens']]
        self.max_length = self.get_max_length()
        print(f'Max Length : {self.max_length}')

        ######## Changed part for negatives ###########
        self.vocab = list([[term] for term in self.token_to_count.keys()])
        self.modifier_vocab = list([[term] for term in self.modifiers_to_count.keys()])
        self.modifier_token_to_id = {self.modifier_vocab[i][0]: i + 1 for i in range(len(self.modifier_vocab))}
        self.modifier_id_to_fit = list([[term] for term in self.modifier_token_to_id.values()])
        ##############      End       #################

        print(f'Vocab Size : {len(self.vocab)}')
        print(f'Modifier Size : {len(self.modifier_vocab)}')
        # print(self.token_to_count)

    def get_max_length(self):
        max_length = 0
        for index, row in self.df.iterrows():
            ######## Changed part for negatives ###########
            token_list = [x for x in row['tokens'] if x not in self.modifiers]
            ##############      End       #################
            tmp_length = len(token_list)
            if tmp_length > max_length:
                max_length = tmp_length
        return max_length

    def k_fold_partition(self, fold_num = 10):
        batch_size = int(len(self.train) / fold_num) # the number of data for each fold
        remain_num = len(self.train) - batch_size * fold_num # the remain data after partition
        self.fold_data = []
        fold_batch_list = [] # Average the remaining data to the folds
        for fold in range(fold_num):
          if remain_num > 0:
            remain_num -= 1
            fold_batch_list.append(batch_size + 1)
          else:
            fold_batch_list.append(batch_size)
        fold_index = 0 # The starting position of each division data
        for fold in range(fold_num):
          fold_texts = [fold_index, fold_index + fold_batch_list[fold]]
          self.fold_data.append(fold_texts)
          # print(f'fold_data : {fold_batch_list[fold]}')
          # print(f'fold_data type : {type(fold_batch_list[fold])}')
          # print(f'index type : {type(fold_index)}')
          fold_index = fold_index + fold_batch_list[fold]


    def k_fold_split(self, k):
        print(f'K : {k}, fold_data[k] : {self.fold_data[k]}')
        validation = self.train.iloc[self.fold_data[k][0]: self.fold_data[k][1]]
        train = self.train.iloc[0: self.fold_data[k][0]].append(self.train.iloc[self.fold_data[k][1]:])
        train_num = len(self.train) - (self.fold_data[k][1] - self.fold_data[k][0])
        if train_num % self.batch_size == 0:
            self.no_batch = train_num / self.batch_size
        else:
            self.no_batch = int(train_num / self.batch_size) + 1
        print(f' index : {self.no_batch}')
        return train, validation



    def split(self, train, test):
        index = int(train * len(self.df))
        self.train = self.df.iloc[0:index]
        self.test = self.df.iloc[index:]
        # if self.train_index % self.batch_size == 0:
        #     self.no_batch = self.train_index / self.batch_size
        # else:
        #     self.no_batch = int(self.train_index / self.batch_size) + 1
        # print(f' index : {self.no_batch}')

    def build_training_data(self, enc, df, is_exp):
        X = []
        Y = []
        #---------------
        metadata = []
        metadata_word_ids = []
        #---------------
        for index, row in df.iterrows():
            # build modifiers
            np_modifier = np.zeros(len(row['tokens']))
            #---------------
            word_ids = []

            if is_exp: 
                self.graph_handler.add_sentence(row['dataset_id'], row['file_id'], row['sentence_id'], row['sentence'], row['label'])
            #---------------
            for index, value in enumerate(row['tokens']):
                # label the word influenced by modifiers
                if value in self.modifiers:
                    #---------------
                    if is_exp:
                        self.graph_handler.add_word(row['dataset_id'], row['file_id'], row['sentence_id'], index, value, True)
                    #---------------
                    id = self.modifier_token_to_id[value]
                    np_modifier[index] = -1
                    for i in range(1, self.context_window + 1):
                        if index - i > 0:
                            np_modifier[index - i] = id
                            if row['tokens'][index - i] in self.modifiers:
                              self.count += 1
                        if index + i < len(row['tokens']):
                            np_modifier[index + i] = id
                            if row['tokens'][index + i] in self.modifiers:
                              self.count += 1
                #---------------
                else:
                    word_ids.append(index)
                    if is_exp:
                        self.graph_handler.add_word(row['dataset_id'], row['file_id'], row['sentence_id'], index, value, False)
                #---------------
                    # np_modifier[index - self.context_window] = id
                    # np_modifier[index] = -1
                    # if (index + self.context_window) < len(row['tokens']):
                    #     np_modifier[index + self.context_window] = id

            ######## Changed part for negatives ###########
            # delete modifier itself
            deleted_index = [i for i in range(len(np_modifier)) if np_modifier[i] == -1]
            np_modifier = np.delete(np_modifier, deleted_index)


            tmp = [enc.transform([[t]]).toarray()[0] for t in row['tokens'] if t not in self.modifiers]
            for i in range(len(tmp)):
                tmp[i] = np.append(tmp[i], np_modifier[i])
            if len(tmp) < self.max_length:
                pad_length = self.max_length - len(tmp)
                for i in range(pad_length):
                    tmp.append(np.append(np.zeros(len(self.vocab)), -1))
                    #---------------
                    word_ids.append(-1)
                    #---------------
            ##############      End       #################
            X.append(tmp)
            Y.append(row['label'])
            #---------------
            metadata.append([row['dataset_id'], row['file_id'], row['sentence_id']])
            metadata_word_ids.append(word_ids)
            #---------------
        X = np.array(X)
        Y = np.array(Y)
        #---------------
        metadata = np.array(metadata)
        metadata_word_ids = np.array(metadata_word_ids)
        #---------------
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        #---------------
        metadata = torch.from_numpy(metadata).int()
        metadata_word_ids = torch.from_numpy(metadata_word_ids).int()
        #---------------
        print(f'Input Data Shape (sequence_num, sequence_len, vocab_size + 1) : {X.shape}')
        print(f'Input Label Shape : {Y.shape}')
        return X, Y, metadata, metadata_word_ids


