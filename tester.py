# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import numpy as np
from dataset import Dataset
from scripts import shredFacts
from scripts import shredTriangle
from de_triangle import DE_Triangle

from measure import Measure

class Tester:
    def __init__(self, dataset, model_path, valid_or_test):
        self.model = torch.load(model_path)
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.tri_l_len = 0
        
    def getRank(self, sim_scores):#assuming the test fact is the first one
        n = 1
        s = 0
        for i in range(self.tri_l_len):
            s = s + sim_scores[i]

        for i in range(self.dataset.numRel()):
            s_t = 0
            for j in range(self.tri_l_len):
                s_t = s_t + sim_scores[(i+1)*self.tri_l_len+j]
            if s_t > s:
                n = n + 1
        return n
    
    def replaceAndShred(self, tri_l, raw_or_fil):

        n = len(tri_l)
        ret_tri = [tri_l for i in range(self.dataset.numRel()+1)]
        ret_tri = np.array(ret_tri).reshape(-1,10)
        ret_tri = [tuple(tri) for tri in ret_tri]
        for id in range(self.dataset.numRel()):
            for i in range (n):
                r1,r2,r3,y,m,d,p2,p3,e1,e2 = ret_tri[n*(id+1)+i]
                ret_tri[n * (id + 1) + i] = (r1,r2,id,y,m,d,p2,p3,e1,e2)
        return shredTriangle(np.array(ret_tri))

    def replaceAndShred1(self, tri_l, raw_or_fil, head_or_tail):
        n = len(tri_l)
        ret_tri = [tri_l for i in range(self.dataset.numEnt() + 1)]
        ret_tri = np.array(ret_tri).reshape(-1, 10)
        ret_tri = [tuple(tri) for tri in ret_tri]
        for id in range(self.dataset.numEnt()):
            for i in range(n):
                r1, r2, r3, y, m, d, p2, p3, e1, e2 = ret_tri[n * (id + 1) + i]
                if head_or_tail == "head":
                    ret_tri[n * (id + 1) + i] = (r1, r2, r3, y, m, d, p2, p3, id, e2)

                if head_or_tail == "tail":
                    ret_tri[n * (id + 1) + i] = (r1, r2, r3, y, m, d, p2, p3, e1, id)
        return shredTriangle(np.array(ret_tri))
    
    def test(self):
        for i, tri_l in enumerate(self.dataset.data_triangle[self.valid_or_test]):
            settings = ["fil"]
            self.tri_l_len = len(tri_l)
            for raw_or_fil in settings:
                for head_or_tail in ["head", "tail"]:
                   r1, r2, r3, years, months, days, p2, p3, e1, e2 = self.replaceAndShred1(tri_l, raw_or_fil,head_or_tail)
                   sim_scores = self.model(r1, r2, r3, years, months, days, p2, p3, e1, e2).cpu().data.numpy()
                   rank = self.getRank(sim_scores)
                   self.measure.update(rank, raw_or_fil)


        self.measure.print_()
        print("~~~~~~~~~~~~~")
        self.measure.normalize(len(self.dataset.data[self.valid_or_test]))
        self.measure.print_()
        
        return self.measure.mrr["fil"]
        
