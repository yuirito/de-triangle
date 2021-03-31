# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import copy
import time
import numpy as np
from random import shuffle
import pickle
from scripts import shredFacts

class Dataset:
    """Implements the specified dataloader"""
    def __init__(self, 
                 ds_name,tri_path,tri_load):
        """
        Params:
                ds_name : name of the dataset 
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.relfrq = {}
        self.triangle = {}
        self.data_triangle = {}
        self.triangle_sorted = []
        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test":  self.readFile(self.ds_path + "test.txt")}
        self.data_triangle["train"] = []
        #for k,v in sorted(self.relfrq.items(), key=lambda kv: (kv[1], kv[0]),reverse=True):
            #print(k,v)
        self.start_batch = 0
        self.all_facts_as_tuples = None
        
        self.convertTimes()
        #print(self.data["train"][0][3],self.data["train"][0][4],self.data["train"][0][5])
        self.ent_list_h = [None for _ in range(self.numEnt()+1)]
        self.ent_list_t = [None for _ in range(self.numEnt() + 1)]
        #self.saveID(self.ds_path + "ent2id.txt",self.ent2id)
        #self.saveID(self.ds_path + "rel2id.txt", self.rel2id)
        if tri_load:
            with open(self.ds_path + tri_path,"rb") as f:
                self.triangle = pickle.load(f)
                self.saveTriangle(self.ds_path + "triangle.txt")
        else:
            self.entGraph()
            self.findTriangle()
            with open(self.ds_path + tri_path, "wb") as f:
                pickle.dump(self.triangle,f)
            self.saveTriangle(self.ds_path + "triangle.txt")
        print("done")
        self.triangle_sorted = [k for (k,v) in sorted(self.triangle.items(),key=lambda kv: (kv[1], kv[0]),reverse=True)]

        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])
        
        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl])
        
    def readFile(self, 
                 filename):

        with open(filename, "r") as f:
            data = f.readlines()
        
        facts = []
        for line in data:
            elements = line.strip().split("\t")
            
            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            timestamp = elements[3]
            
            facts.append([head_id, rel_id, tail_id, timestamp])
            
        return facts
    
    
    def convertTimes(self):      
        """
        This function spits the timestamp in the day,date and time.
        """  
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date
                
                
    def entGraph(self):
        for split in ["train"]:
            for i, fact in enumerate(self.data[split]):
                h = fact[0]
                r = fact[1]
                t = fact[2]
                y = fact[3]
                m = fact[4]
                d = fact[5]
                if self.ent_list_h[h] == None:
                    self.ent_list_h[h] = [(r, t, y, m, d)]
                else:
                    self.ent_list_h[h].append((r, t, y, m, d))
                if self.ent_list_t[t] == None:
                    self.ent_list_t[t] = [(r, h, y, m ,d)]
                else:
                    self.ent_list_t[t].append((r, h, y, m, d))


    def findTriangle(self):
        for e1,rel_list in enumerate(self.ent_list_h):
            print(e1)
            if rel_list != None:
                for j,rel_tuple in enumerate(rel_list):
                    (r1,e2,y1,m1,d1) = rel_tuple
                    if self.ent_list_h[e2] == None:
                        continue
                    else:
                        for k, rel_tuple2 in enumerate(self.ent_list_h[e2]):
                            (r2,e3,y2,m2,d2) = rel_tuple2
                            if self.ent_list_h[e3] == None:
                                continue
                            else:
                                for m,rel_tuple3 in enumerate(self.ent_list_h[e3]):
                                    (r3,e,y3,m3,d3) = rel_tuple3
                                    if e == e1:
                                        rel_list = [r1, r2, r3]
                                        rel_list.sort()
                                        rel_triangle = (rel_list[0], rel_list[1], rel_list[2], 0)
                                        if rel_triangle in self.triangle:
                                            self.triangle[rel_triangle] = self.triangle[rel_triangle] + 1
                                            #print ("%d :" % e1)
                                            #print("{}:{}".format(rel_triangle,self.triangle[rel_triangle]))
                                        else:
                                            #print ("%d :" % e1)
                                            self.triangle[rel_triangle] = 1
                                            self.data_triangle["train"].append(rel_triangle)


        for e1, rel_list in enumerate(self.ent_list_t):
            print(e1)
            if rel_list != None:
                for j, rel_tuple in enumerate(rel_list):
                    (r1, e2, y1, m1, d1) = rel_tuple
                    if self.ent_list_h[e2] == None:
                        continue
                    else:
                        for k, rel_tuple2 in enumerate(self.ent_list_h[e2]):
                            (r2, e3, y2, m2, d2) = rel_tuple2
                            if self.ent_list_h[e3] == None:
                                continue
                            else:
                                for m, rel_tuple3 in enumerate(self.ent_list_h[e3]):
                                    (r3, e, y3, m3, d3) = rel_tuple3
                                    if e == e1:
                                        rel_triangle = (r1,r2,r3,1)
                                        if rel_triangle in self.triangle:
                                            self.triangle[rel_triangle] = self.triangle[rel_triangle] + 1
                                            # print ("%d :" % e1)
                                            # print("{}:{}".format(rel_triangle,self.triangle[rel_triangle]))
                                        else:
                                            # print ("%d :" % e1)
                                            self.triangle[rel_triangle] = 1
                                            self.data_triangle["train"].append(rel_triangle)


    def saveTriangle(self,filename):
        with open(filename, "w") as f:
            for k,v in sorted(self.triangle.items(),key=lambda kv: (kv[1], kv[0]),reverse=True):
                f.write("{} {}\n".format(k,v))


    def saveID(self,filename,dict):
        with open(filename, "w") as f:
            for k,v in sorted(dict.items(), key=lambda kv: (kv[1], kv[0])):
                f.write("%d %s\n" %(v,k))


    #def find_potential_triangle_top3(self):



    def numEnt(self):
    
        return len(self.ent2id)

    def numRel(self):
    
        return len(self.rel2id)

    
    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name] 
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]
    
    def getRelID(self, rel_name):

        if rel_name in self.rel2id:
            self.relfrq[rel_name] = self.relfrq[rel_name] + 1
            return self.rel2id[rel_name] 
        self.rel2id[rel_name] = len(self.rel2id)
        self.relfrq[rel_name] = 1
        #print(rel_name)
        return self.rel2id[rel_name]

    
    def nextPosBatch(self, batch_size):
        if self.start_batch + batch_size > len(self.data["train"]):
            ret_facts = self.data["train"][self.start_batch : ]
            self.start_batch = 0
        else:
            ret_facts = self.data["train"][self.start_batch : self.start_batch + batch_size]
            self.start_batch += batch_size
        return ret_facts
    

    def addNegFacts(self, bp_facts, neg_ratio):
        ex_per_pos = 2 * neg_ratio + 2
        facts = np.repeat(np.copy(bp_facts), ex_per_pos, axis=0)
        for i in range(bp_facts.shape[0]):
            s1 = i * ex_per_pos + 1
            e1 = s1 + neg_ratio
            s2 = e1 + 1
            e2 = s2 + neg_ratio
            
            facts[s1:e1,0] = (facts[s1:e1,0] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            facts[s2:e2,2] = (facts[s2:e2,2] + np.random.randint(low=1, high=self.numEnt(), size=neg_ratio)) % self.numEnt()
            
        return facts
    
    def addNegFacts2(self, bp_facts, neg_ratio):
        pos_neg_group_size = 1 + neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.numEnt(), size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.numEnt(), size=facts2.shape[0])
        
        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0
        
        facts1[:,0] = (facts1[:,0] + rand_nums1) % self.numEnt()
        facts2[:,2] = (facts2[:,2] + rand_nums2) % self.numEnt()
        return np.concatenate((facts1, facts2), axis=0)
    
    def nextBatch(self, batch_size, neg_ratio=1):
        bp_facts = self.nextPosBatch(batch_size)
        batch = shredFacts(self.addNegFacts2(bp_facts, neg_ratio))
        return batch
    
    
    def wasLastBatch(self):
        return (self.start_batch == 0)
            
