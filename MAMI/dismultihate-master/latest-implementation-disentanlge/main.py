import torch
import torch.nn as nn
from torch.utils.data import Subset,ConcatDataset

from dataset import Wraped_Data
from train import train_for_deep
import utils
import config
import os
import pickle as pkl

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    train_set=Wraped_Data(opt,'train')
    
    val_set=None
    off_val=None
    off_test=None
    
    if opt.DATASET=='mem':
        test_set=Wraped_Data(opt,'dev_seen')
        if opt.OFF_EVAL:
            off_val=Wraped_Data(opt,'train',pretraining=True)
            off_test=Wraped_Data(opt,'test',pretraining=True)
    elif opt.DATASET=='mmhs':
        val_set=Wraped_Data(opt,'val')
        print ('Length of test:',len(val_set))
        test_set=Wraped_Data(opt,'test')
    elif opt.DATASET=='off':
        val_set=Wraped_Data(opt,'val')
        test_set=Wraped_Data(opt,'test')
    print ('Length of train:',len(train_set))
    print ('Length of test:',len(test_set))
    
    if opt.MODEL=='bert':
        from transformers import BertForSequenceClassification,BertConfig
        if opt.DATASET in ['mem','mmhs','off']:
            final_dim=1
        model=BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=final_dim,
            output_attentions=False,
            output_hidden_states=False
        )    
        model=model.cuda()
    elif opt.MODEL=='latent':
        import latent_reason
        constructor='build_baseline'
        model=getattr(latent_reason,constructor)(train_set,opt).cuda()
    
    train_for_deep(model,train_set,test_set,opt,
                   val_set,
                   off_val,off_test)
    exit(0)
    