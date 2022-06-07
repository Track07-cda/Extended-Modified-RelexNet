import argparse
import json
import os
import torch.optim

from relexnet import RelexNet
# from models import FastText, LSTM
from training import train_model, test_model
from sklearn.utils import shuffle
import os
import glob
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from data_loader import DataLoader
from collections import Counter, defaultdict
from graphHandler import GraphHandler

parser = argparse.ArgumentParser(description='Relexnet with modifier')

parser.add_argument('--config', type=str, help='Path to config file')
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

config['mode'] = list(map(str.lower, config['mode']))

if 'model' in config.keys():
    if 'epochNum' not in config['model'].keys():
        config['model']['epochNum'] = 20
    if 'batchSize' not in config['model'].keys():
        config['model']['batchSize'] = 32
    if 'contextWindowSize' not in config['model'].keys():
        config['model']['contextWindowSize'] = 1
    if 'kFoldSplit' not in config['model'].keys():
        config['model']['kFoldSplit'] = 2
    if 'classNum' not in config['model'].keys():
        config['model']['classNum'] = 2
    if 'paddingLength' not in config['model'].keys():
        config['model']['paddingLength'] = 70
    if 'foldNum' not in config['model'].keys():
        config['model']['foldNum'] = 10
    if 'testPortion' not in config['model'].keys():
        config['model']['testPortion'] = 0.1 if 'test' in config['mode'] else 0

graph_handler=GraphHandler(config['ontology']['path'], config['ontology']['ontoPrefix'], config['ontology']['dataPrefix'], config['model'])

num_epochs = 20
batch_size = 32  # Number of hidden neurons in model
context_window = 1
k = 9

dev = 'cuda' if torch.cuda.is_available() else 'cpu'  # If you have a GPU installed, use that, otherwise CPU
print(dev)
print('Loading data...')
data = pd.DataFrame()

dataset_list = dict()
for i in range(len(config['data'])):
    data_f = pd.read_csv(config['data'][i]['path'], header=None,
                         index_col=None, names=['sentence', 'label'])
    dataset_name = config['data'][i]['dataset']
    if dataset_name not in dataset_list.keys():
        dataset_list[dataset_name] = len(dataset_list)
        graph_handler.add_dataset(len(dataset_list) - 1, dataset_name)

    data_f['dataset_id'] = dataset_list[dataset_name]
    data_f['file_id'] = i
    data_f['sentence_id'] = data_f.index
    data = data.append(data_f, ignore_index=True)
    graph_handler.add_file(dataset_list[dataset_name], i, config['data'][i]['path'])

data = data.sample(frac=1, random_state=1)
# data = shuffle(data)
enc_modifiers = OneHotEncoder()
enc_tokens = OneHotEncoder()

dataset = DataLoader(data, config=config, graph_handler=graph_handler, dev=dev, enc_modifiers=enc_modifiers, enc_tokens=enc_tokens)
# print(f'Data Size: {dataset.df.size()}')
print("Data Ready!")
vocabulary_size = len(dataset.vocab)
len_seq = dataset.max_length

# model = FastText(len(dataset.token_to_id)+2, num_hidden, len(dataset.class_to_id)).to(dev)
model = RelexNet(vocabulary_size=vocabulary_size, enc_tokens=dataset.enc_tokens, enc_modifiers=dataset.enc_modifiers, modifier_size=len(
    dataset.modifier_vocab), len_seq=len_seq, num_classes=config['model']['classNum'], graph_handler=graph_handler).cuda()

if 'train' in config['mode']:
    print('Start training!')
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.9,0.99), weight_decay=10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
                                                    8 * dataset.no_batch, 14 * dataset.no_batch], gamma=0.1, last_epoch=-1)
    losses, accuracies = train_model(
        dataset, model, optimizer, scheduler, config['model']['epochNum'], dev=dev)
    print(losses)

if 'path' in config['model']:
    model.load_state_dict(torch.load(config['model']['path']))
    model = model.cuda()
    print('Model loaded')

if 'test' in config['mode']:
    print('Start testing!')
    test_accuracy, output = test_model(dataset, model, dev)
    print(test_accuracy)
    print(output)

if 'output' in config.keys():
    if 'explanation' in config['output'].keys():
        graph_handler.graph.serialize(
            destination=config['output']['explanation'], format='turtle')
    if 'model' in config['output'].keys():
        torch.save(model.state_dict(), os.path.join(config['output']['model']))
