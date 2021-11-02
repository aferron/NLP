import os
import pandas as pd
import re
import json
import pickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
import config
import itertools
import random
import string
from nltk.tokenize.treebank import TreebankWordTokenizer
from preprocessing import clean_text

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Wraped_Data():
    def __init__(self,opt,mode='train',pretraining=False):
        super(Wraped_Data,self).__init__()
        self.opt=config.parse_opt()
        if pretraining:
            self.dataset='off'
        else:
            self.dataset=opt.DATASET
        self.mode=mode
        
        if opt.DEBUG:
            self.entries=self.load_tr_val_entries()[:100]
        else:
            self.entries=self.load_tr_val_entries()
        
        print ('The length of the dataset for:',mode,'is:',len(self.entries))
        if opt.MODEL=='bert' or opt.MODEL=='latent':
            from transformers import BertTokenizer
            self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
   
    def load_tr_val_entries(self):
        if self.dataset=='mem':
            all_data=read_json(os.path.join(self.opt.DATA,
                                            self.mode+'_external.json'))
            entity=load_pkl(os.path.join(self.opt.DATA,
                                         'winner-info/annotations/entity_dict.pkl'))
            race=load_pkl(os.path.join(self.opt.DATA,
                                       'winner-info/annotations/race_dict.pkl'))
        elif self.dataset=='mmhs':
            all_data=read_json(os.path.join(self.opt.MM,
                                            self.mode+'_external.json'))
            entity=load_pkl(os.path.join(self.opt.MM,
                                         'entity_dict.pkl'))
            race=load_pkl(os.path.join(self.opt.MM,
                                       'race_dict.pkl'))
        elif self.dataset=='off':
            all_data=read_json(os.path.join(self.opt.OFF,
                                            self.mode+'_external.json'))
            entity=load_pkl(os.path.join(self.opt.OFF,
                                         'entity_dict.pkl'))
            race=load_pkl(os.path.join(self.opt.OFF,
                                       'race_dict.pkl'))
        entries=[]
        for info in all_data:
            sent=info['text']
            if self.dataset=='mem':
                if self.mode in ['train','dev_seen','dev_unseen']:
                    label=info['label']
                else:
                    label=0
                img_id=info['img'].split('/')[1]
            elif self.dataset=='mmhs' or self.dataset=='off':
                label=info['label']
                img_id=info['img']
                
            cur_race=race[img_id]
            cur_entity=entity[img_id]
            
            entry={
                'text':sent,
                'img_id':img_id,
                'race':cur_race,
                'entity':cur_entity,
                'answer':label
            }
            entries.append(entry)
        return entries
    
    def padding_sent(self,tokens,length,mode='normal'):
        l_c=len(tokens)
        if len(tokens)<length:
            if mode=='normal':
                padding=[self.dictionary.ntokens-1]*(length-len(tokens))
                tokens=padding+tokens
            else:
                padding=[0]*(length-len(tokens))
                tokens=tokens+padding
        else:
            l_c=length
            tokens=tokens[:length]
        return tokens,l_c
   
    def get_list(self,entity):
        result=[]
        for e in entity:
            if len(e)>0:
                result.extend(e)
        if len(result)==0:
            result.append('unk')
        return result
    
    def extract_list(self,race):
        result=[]
        for info in race:
            if 'race' in info.keys() and info['race'] is not None:
                result.append(info['race'])
            if 'gender' in info.keys() and info['gender'] is not None:
                result.append(info['gender'])
        if len(result)==0:
            result.append('unk')
        return result
            
    
    def generate_bert_info(self,race_list,entity_list,text):
        type_tokens=[]
        if type(text) is not str:
            text=''
        total=text+' [SEP] '+' '.join(entity_list)+' [SEP] '+' '.join(race_list)
        external=' '.join(entity_list)+' [SEP] '+' '.join(race_list)
        tokens=self.tokenizer.encode(
                total,
                add_special_tokens=True, # add [CLS] and [SEP]
                max_length=64,
                truncation=True
            )
        external_tokens=self.tokenizer.encode(
                total,
                add_special_tokens=True, # add [CLS] and [SEP]
                max_length=32,
                truncation=True
            )
        #print (total,tokens)
        start=0
        pad_tokens,_=self.padding_sent(tokens,64,'bert')
        external_pad,_=self.padding_sent(external_tokens,32,'bert')
        flag=True
        for t in pad_tokens:
            type_tokens.append(start)
            if t==102 and flag:
                start+=1
                flag=False
        return pad_tokens,type_tokens,external_pad
    
    def __getitem__(self,index):
        entry=self.entries[index]
        vid=entry['img_id']
        
        if self.dataset in ['mem','mmhs','off']:
            label=torch.tensor(entry['answer'])
        else:
            label=torch.from_numpy(entry['answer'])
        
        #feat=np.array(self.feat_file[vid],dtype=np.float32)
        if self.dataset=='mem':
            feat=np.load(os.path.join(self.opt.DATA,'faster_hatefulmem_clean_36',
                                  vid.split('.')[0]+'.npy'),
                     allow_pickle=True).item()['features']
        elif self.dataset=='mmhs':
            feat=np.load(os.path.join(self.opt.MM,
                                      self.mode+'_features',
                                      vid.split('.')[0]+'.npy'),
                     allow_pickle=True).item()['features']
        elif self.dataset=='off':
            feat=np.load(os.path.join(self.opt.OFF,
                                      'littleBoy',
                                      'clean_features',
                                      vid.split('.')[0]+'.npy'),
                     allow_pickle=True).item()['features']
        feat=torch.from_numpy(feat)
        
        entity=entry['entity']
        entity_list=self.get_list(entity)
        
        race=entry['race']
        race_list=self.extract_list(race)
        
        batch={
            'img_id':vid,
            'feature':feat,
            'answer':label
        }
        if self.opt.MODEL=='bert' or self.opt.MODEL=='latent':
            pad_tokens,type_tokens,external_pad= \
            self.generate_bert_info(race_list,entity_list,entry['text'])
            #print (len(pad_tokens))
            batch['bert']=torch.from_numpy(np.array(pad_tokens,dtype=np.int64))
            batch['ext']=torch.from_numpy(np.array(external_pad,dtype=np.int64))
            mask=[int(num>0) for num in pad_tokens]
            ext_mask=[int(num>0) for num in external_pad]
            #print (len(mask),len(type_tokens))
            batch['mask']=torch.from_numpy(np.array(mask,dtype=np.int64))
            batch['ext_mask']=torch.from_numpy(np.array(ext_mask,dtype=np.int64))
            batch['type_token']=torch.LongTensor(type_tokens)
        return batch
        
        
    def __len__(self):
        return len(self.entries)
    
