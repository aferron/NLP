"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
import torch.nn.functional as F
import math

class Basic_Attention(nn.Module):
    def __init__(self,hidden,mid,dropout):
        super(Basic_Attention,self).__init__()
        self.v_proj=nn.Linear(hidden,mid)
        self.q_proj=nn.Linear(hidden,mid)
        self.vlinear=nn.Linear(hidden,hidden)
        self.att=nn.Linear(mid,1)
        self.softmax=nn.Softmax(dim=1)
        self.dropout=nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden)
        self.relu = nn.ReLU()


    def forward(self,v,q):
        
        v_proj=self.v_proj(v)
        v_proj=self.dropout(v_proj)
        
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        q_proj=self.dropout(q_proj)
        
        vq_proj=F.relu(v_proj +q_proj)
        proj=self.att(vq_proj)#B,L,1
        w_att=self.softmax(proj)
        
        vatt=torch.sum(v * w_att,dim=1)
        
        return vatt


class Gate_Attention(nn.Module):
    def __init__(self,num_hidden):
        super(Gate_Attention,self).__init__()
        self.hidden=num_hidden
        self.w1=nn.Parameter(torch.Tensor(num_hidden,num_hidden))
        self.w2=nn.Parameter(torch.Tensor(num_hidden,num_hidden))
        self.bias=nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()
        
    def reset_parameter(self):
        stdv1=1. / math.sqrt(self.hidden)
        stdv2=1. / math.sqrt(self.hidden)
        stdv= (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1,stdv1)
        self.w2.data.uniform_(-stdv2,stdv2)
        self.bias.data.uniform_(-stdv,stdv)
        
    def forward(self,a,b):
        wa=torch.matmul(a,self.w1)
        wb=torch.matmul(b,self.w2)
        gated=wa+wb+self.bias
        gate=torch.sigmoid(gated)
        output=gate * a + (1-gate) * b
        return output

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            return p.view(-1, self.glimpse, v_num, q_num), logits

        return logits
