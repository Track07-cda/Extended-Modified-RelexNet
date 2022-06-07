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
    def __init__(self, vocabulary_size, modifier_size, len_seq, num_classes, enc_modifiers, enc_tokens, graph_handler):
        super(RelexNet, self).__init__()
        self.no_modifiers = np.zeros(modifier_size)
        self.num_classes = num_classes
        self.L = nn.Linear(vocabulary_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.M = nn.Linear(modifier_size, 1).cuda()
        # self.B = nn.Linear(len_seq, len_seq)
        self.enc_modifiers = enc_modifiers
        self.enc_tokens = enc_tokens
        self.softmax = nn.Softmax(dim=1)
        self.explain = False
        self.gh = graph_handler


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
                modifier.append(self.enc_modifiers.transform([[modifier_ids[i].cpu()]]).toarray()[0])
                influence_flag.append(1)
        modifier = torch.tensor(modifier).float().cuda()
        influence_flag = torch.tensor(influence_flag).float()
        influence_flag = influence_flag.reshape(-1, 1).cuda()
        add = add.cuda()
        return modifier, influence_flag, add


    def forward(self, input, meta, meta_wi, B_layer):
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

            S_layer = L_onestep * modifier_score

            B_layer_in = B_layer.detach()
            B_layer = torch.add(B_layer, S_layer)

            if self.num_classes == 1:
                O_layer[t] = F.sigmoid(B_layer)
            else:
                O_layer = self.softmax(B_layer)

            if self.explain:
                for index in range(len(meta)):
                    m = meta[index]
                    wi = meta_wi[index][t]
                    if wi != -1:
                        for class_id in range(self.num_classes):
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerL', None , float(L_onestep[index][class_id]), None)
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerM', float(modifier_ids[index]), float(modifier_score[index][0]), None)
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerS', float(L_onestep[index][class_id]), float(S_layer[index][class_id]), 'layerL')
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerS', float(modifier_score[index][0]), float(S_layer[index][class_id]), 'layerM')
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerB', float(S_layer[index][class_id]), float(B_layer[index][class_id]), 'layerS')
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerB', float(B_layer_in[index][class_id]), float(B_layer[index][class_id]), 'layerB')
                            self.gh.add_evaluation(m[0], m[1], m[2], wi, class_id, 'layerO', float(B_layer[index][class_id]), float(O_layer[index][class_id]), 'layerB') 

        return O_layer, B_layer

    def enable_explain(self):
        self.explain = True

    def disable_explain(self):
        self.explain = False
