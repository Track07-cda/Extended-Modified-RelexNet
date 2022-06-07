import os
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

from torch.autograd import Variable

class RelexNet(nn.Module):
    #Single step RNN.
    #input_size is char_vacab_size=26,hidden_size is number of hidden neurons，output_size is number of categories
    def __init__(self, vocabulary_size, modifier_size, len_seq, num_classes, enc):
        super(RelexNet, self).__init__()
        self.no_modifiers = np.zeros(modifier_size)
        self.num_classes = num_classes
        self.L = nn.Linear(vocabulary_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.M = nn.Linear(modifier_size, 1).cuda()
        # self.B = nn.Linear(len_seq, len_seq)
        self.enc = enc
        self.softmax = nn.Softmax(dim=1)


    def handle_modifier(self, modifier_ids, batch):
        # modifier_id = modifier_ids.cpu()
        modifier = []
        influence_flag = []
        add = 0.5 * Variable(torch.ones(batch, 1))
        for i in range(batch):
            if modifier_ids[i] == 0:
                modifier.append(self.no_modifiers)
                influence_flag.append(0)
                add[i] = 1
            elif modifier_ids[i] == -1:
                modifier.append(self.no_modifiers)
                influence_flag.append(0)
                add[i] = 0
            else:
                # print(modifier_ids[i])
                modifier.append(self.enc.transform([[modifier_ids[i].cpu()]]).toarray()[0])
                influence_flag.append(1)
        modifier = torch.tensor(modifier).float().cuda()
        influence_flag = torch.tensor(influence_flag).float()
        influence_flag = influence_flag.reshape(-1, 1).cuda()
        add = add.cuda()
        return modifier, influence_flag, add



    def forward(self, input, B_layer):
        # X.shape = (batch, seq_len, vocab_size)
        T = input.shape[1]
        batch = input.shape[0]
        predict_y = Variable(torch.zeros(batch, self.num_classes))

        if B_layer is None:
            B_layer = Variable(torch.zeros(batch, self.num_classes)).cuda()

        for t in range(T):
            tmp = input[:, t, :-1]
            modifier_ids = input[:, t, -1]
            modifier, influence_flag, add = self.handle_modifier(modifier_ids, batch)

            L_onestep = self.L(tmp)
            L_onestep = self.dropout(L_onestep)
            # L_onestep = torch.sigmoid(L_onestep)
            # L_onestep = MyReLU_PLUS.apply(L_onestep)
            L_onestep = F.relu6(L_onestep)
            modifier_score = torch.sigmoid(self.M(modifier))
            modifier_score = modifier_score * influence_flag
            modifier_score = modifier_score + add
            # L_onestep = L_onestep + modifier_score

            L_onestep = L_onestep * modifier_score
            # L_onestep = L_onestep * negative_score

            B_layer = torch.add(B_layer, L_onestep)
            # print(B_layer)

            if self.num_classes == 1:
                predict_y[t] = F.sigmoid(B_layer)
            else:
                predict_y = self.softmax(B_layer)

        return predict_y, B_layer
