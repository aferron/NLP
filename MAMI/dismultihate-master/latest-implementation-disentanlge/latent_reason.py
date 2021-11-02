from transformers import BertForSequenceClassification,BertConfig
from rela_encoder import Rela_Module
import torch.nn as nn
import torch
from torch.nn import functional
from attention import Basic_Attention
from classifier import SingleClassifier, SimpleClassifier

def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1).cuda()
    return one_hot

def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = functional.softmax(logits, dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs

def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y

class Latent_Bert(nn.Module):
    def __init__(self,text_bert,multimodal_fusion,linear,
                 fc_gt,fc_a,scorer,
                 v_dim,temp):
        super(Latent_Bert,self).__init__()
        
        self.text_bert=text_bert
        self.transfomer=multimodal_fusion
        self.softmax=nn.Softmax(dim=1)
        
        self.v_dim=v_dim
        self.temp=temp
        
        self.fc_gt=fc_gt
        self.fc_a=fc_a
        self.scorer=scorer
        
        self.linear=linear
        
    def forward(self,v,text,ext,token_type_ids,mask,ext_mask):
        #print (v.shape,self.v_dim)
        v=self.linear(v)
        bert_out=self.text_bert(text,
                                token_type_ids=token_type_ids,
                                attention_mask=mask)
        ext_repre=self.text_bert(ext,
                                token_type_ids=None,
                                attention_mask=ext_mask)[1][-1][:,0]
        
        text_repre=bert_out[1][-1]
        align_repre=self.transfomer(v,text_repre)
        
        gt_target=st_gumbel_softmax(self.softmax(self.fc_gt(ext_repre)),self.temp)
        align_target=self.softmax(self.fc_a(align_repre))
        #print (gt_target.shape,align_target.shape,text_repre[:,0].shape)
        score=torch.sigmoid(self.scorer(
            torch.cat((align_repre,text_repre[:,0]),
                      dim=1)))
        return score,gt_target,align_target
        
def build_baseline(dataset,opt): 
    text_bert=BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1,
        output_attentions=False,
        output_hidden_states=True
    )
    if opt.DATASET in ['mem','mmhs','off']:
        final_dim=1
    multimodal_fusion=Rela_Module(768,
                                  768,opt.NUM_HEAD,opt.MID_DIM,
                                  opt.TRANS_LAYER,
                                  opt.FC_DROPOUT)
    fc_gt=SingleClassifier(768,opt.NUM_LATENT,opt.FC_DROPOUT)
    fc_a=SingleClassifier(768,opt.NUM_LATENT,opt.FC_DROPOUT)
    scorer=SimpleClassifier(768*2,opt.MID_DIM,final_dim,opt.FC_DROPOUT)
    linear=SingleClassifier(opt.FEAT_FC_DIM,768,opt.FC_DROPOUT)
    return Latent_Bert(text_bert,multimodal_fusion,linear,fc_gt,fc_a,scorer,
                       opt.FEAT_FC_DIM,opt.TEMPERATURE)