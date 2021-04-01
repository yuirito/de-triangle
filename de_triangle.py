# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class DE_Triangle(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_Triangle, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.rel_embs_t      = nn.Embedding(dataset.numRel(), params.s_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.create_time_embedds()
        
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.rel_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def create_time_embedds(self):
            
        self.m_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        self.m_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

    def get_time_embedd(self, rel, year, month, day):
        
        y = self.y_amp(rel)*self.time_nl(self.y_freq(rel)*year + self.y_phi(rel))
        m = self.m_amp(rel)*self.time_nl(self.m_freq(rel)*month + self.m_phi(rel))
        d = self.d_amp(rel)*self.time_nl(self.d_freq(rel)*day + self.d_phi(rel))
        
        return y+m+d

    def getEmbeddings(self, r1, r2, r3, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)

        r1_emb,r2_emb,r3_emb = self.rel_embs(r1), self.rel_embs(r2), self.rel_embs_t(r3)
        
        r3_emb_t = self.get_time_embedd(r3, years, months, days)

        r3_emb = torch.cat((r3_emb,r3_emb_t), 1)

        return r1_emb, r2_emb, r3_emb
    
    def forward(self, r1, r2, r3, years, months, days, p2, p3):
        r1_embs, r2_embs, r3_embs = self.getEmbeddings(r1, r2, r3, years, months, days)
        
        scores = r1_embs + p2 * r2_embs + p3 * r3_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim = 1)
        return scores
        