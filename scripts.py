# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

def shredFacts(facts): #takes a batch of facts and shreds it into its columns
        
    heads      = torch.tensor(facts[:,0]).long().cuda()
    rels       = torch.tensor(facts[:,1]).long().cuda()
    tails      = torch.tensor(facts[:,2]).long().cuda()
    years = torch.tensor(facts[:,3]).float().cuda()
    months = torch.tensor(facts[:,4]).float().cuda()
    days = torch.tensor(facts[:,5]).float().cuda()
    return heads, rels, tails, years, months, days


def shredTriangle(tri):  # takes a batch of facts and shreds it into its columns

    r1 = torch.tensor(tri[:, 0]).long().cuda()
    r2= torch.tensor(tri[:, 1]).long().cuda()
    r3 = torch.tensor(tri[:, 2]).long().cuda()
    years = torch.tensor(tri[:, 3]).float().cuda()
    months = torch.tensor(tri[:, 4]).float().cuda()
    days = torch.tensor(tri[:, 5]).float().cuda()
    p2 = torch.tensor(tri[:, 6]).float().cuda()
    p3 = torch.tensor(tri[:, 7]).float().cuda()
    e1 = torch.tensor(tri[:, 8]).long().cuda()
    e2 = torch.tensor(tri[:, 9]).long().cuda()
    return r1, r2, r3, years, months, days, p2, p3, e1, e2