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

        self.ent_embs = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        
        self.create_time_embedds()
        
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.rel_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_uniform_(self.ent_embs.weight)
        
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

    def get_time_embedd(self, entities, year, month, day):
        y = self.y_amp(entities) * self.time_nl(self.y_freq(entities) * year + self.y_phi(entities))
        m = self.m_amp(entities) * self.time_nl(self.m_freq(entities) * month + self.m_phi(entities))
        d = self.d_amp(entities) * self.time_nl(self.d_freq(entities) * day + self.d_phi(entities))

        return y + m + d

    def getEmbeddings(self, r1, r2, r3, years, months, days,e1,e2, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)

        r1_emb,r2_emb,r3_emb = self.rel_embs(r1), self.rel_embs(r2), self.rel_embs(r3)
        e1_emb,e2_emb = self.ent_embs(e1),self.ent_embs(e2)
        e1_emb_t = self.get_time_embedd(e1,years,months,days)
        e2_emb_t = self.get_time_embedd(e1, years, months, days)

        e1_emb = torch.cat((e1_emb, e1_emb_t), 1)
        e2_emb = torch.cat((e2_emb, e2_emb_t), 1)

        return r1_emb, r2_emb, r3_emb , e1_emb, e2_emb
    
    def forward(self, r1, r2, r3, years, months, days, p2, p3, e1, e2):
        r1_embs, r2_embs, r3_embs,e1_emb,e2_emb = self.getEmbeddings(r1, r2, r3, years, months, days,e1,e2)
        p2 = p2.view(-1,1)
        p3 = p3.view(-1,1)
        p2 = p2.repeat(1,self.params.s_emb_dim+self.params.t_emb_dim)
        p3 = p3.repeat(1,self.params.s_emb_dim+self.params.t_emb_dim)

        scores = self.params.beta*(r1_embs + p2 * r2_embs + p3 * r3_embs) + (1-self.params.beta)*(e1_emb +e2_emb -r3_embs)
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = -torch.norm(scores, dim = 1)


        return scores