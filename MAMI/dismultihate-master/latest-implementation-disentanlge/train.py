import os
import time 
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import config
import numpy as np
import h5py
import pickle as pkl
import json
import optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report,precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

    
def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))

def bce_for_loss(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

def compute_score(logits,labels):
    score=(logits.round()==labels).sum().item()
    return score

def compute_auc_score(logits,label):
    bz=logits.shape[0]
    logits=logits.cpu().numpy()
    label=label.cpu().numpy()
    auc=roc_auc_score(label,logits,average='weighted')*bz
    return auc

def compute_multi_score(logits,labels):
    #print (logits.shape,labels.shape)
    logits=torch.max(logits,1)[1]
    labels=torch.max(labels,1)[1]
    score=logits.eq(labels)
    score=score.sum().float()
    return score

def compute_other(logits,labels,binary=False):
    #label=labels.cpu().numpy()
    #logits=logits.cpu.numpy()
    acc=0.0
    if not binary:
        acc=compute_multi_score(logits,labels,binary)
        logits=np.argmax(logits.cpu().numpy(),axis=1)
        label=np.argmax(labels.cpu().numpy(),axis=1)
    else:
        logits=logits.round().squeeze().cpu().numpy()
        label=labels.squeeze().cpu().numpy()
    #print (logits,label)
    bz=logits.shape[0]
    
    f1=f1_score(label,logits
                ,average='weighted',labels=np.unique(label))*bz
    recall=recall_score(label,logits,
                        average='weighted',labels=np.unique(label))*bz
    precision=precision_score(label,logits,
                              average='weighted',labels=np.unique(label))*bz
    result=precision_recall_fscore_support(label,logits,
                                           beta=1.0, labels=None, 
                                           pos_label=1, average=None)
    #f1=result[2][1]*bz
    #recall=result[1][1]*bz
    #precision=result[0][1]*bz
    return acc,f1,recall,precision

def train_for_deep(model,train_set,test_set,opt,
                   val_set=None,
                   off_val=None, off_test=None):
    optim=optimizer.get_std_opt(model, opt)
    logger=utils.Logger(os.path.join(opt.RESULT,'log'+str(opt.SAVE_NUM)+'.txt'))
    log_hyperpara(logger,opt)
    
    train_size=len(train_set)
    test_size=len(test_set)
    train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    if opt.OFF_EVAL and opt.DATASET=='mem':
        off_val_loader=DataLoader(off_val,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        off_test_loader=DataLoader(off_test,
                                   opt.BATCH_SIZE,shuffle=True,num_workers=1)
    if opt.DATASET=='mmhs' or opt.DATASET=='off':
        val_loader=DataLoader(val_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
    if opt.MODEL=='bert' or opt.MODEL=='latent':
        from transformers import get_linear_schedule_with_warmup,AdamW
        optim=AdamW(model.parameters(),
                        lr=2e-5,
                        eps=1e-8
                       )
        num_training_steps=len(train_loader) * opt.EPOCHS
        scheduler=get_linear_schedule_with_warmup(optim,
                                                  num_warmup_steps=0,
                                                  num_training_steps=num_training_steps
                                                 )
    loss_fn=torch.nn.BCELoss()
    for epoch in range(opt.EPOCHS):
        total_loss=train_score=eval_loss=eval_score=0.0
        t=time.time()
        for i, batch in enumerate(train_loader):
            labels=batch['answer'].float().cuda()
            if opt.DATASET in ['mem','mmhs','off']:
                labels=labels.view(-1,1)
            feat=batch['feature'].float().cuda()
            if opt.MODEL=='bert' or opt.MODEL=='latent':
                bert_tokens=batch['bert'].long().cuda()
                mask=batch['mask'].long().cuda()
                token_type_ids=batch['type_token'].long().cuda()
                #print (type(bert_tokens),type(mask),type(token_type_ids))
                #print (bert_tokens)
                if opt.MODEL=='bert':
                    pred=torch.sigmoid(
                        model(bert_tokens, 
                              token_type_ids=token_type_ids,
                              attention_mask=mask)[0]
                    )
                else:
                    external_tokens=batch['ext'].cuda()
                    ext_mask=batch['ext_mask'].cuda()
                    pred,anno_target,align_target=model(
                        feat,
                        bert_tokens,
                        external_tokens,
                        token_type_ids,
                        mask,
                        ext_mask)
                    
                
            if opt.MODEL=='latent':
                if opt.DATASET in ['mem' ,'mmhs', 'off']:
                    loss_f=loss_fn(pred,labels)
                else:
                    batch_score=compute_multi_score(pred,labels)
                    loss_f=bce_for_loss(pred,labels)
                loss_l=bce_for_loss(align_target,anno_target)
                if opt.ABLATION:
                    loss=loss_f
                else:
                    loss=loss_f+opt.LATENT_RATE*loss_l
            else:
                loss=loss_fn(pred,labels)
            total_loss+=loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step()
            if opt.MODEL=='bert':
                scheduler.step()
            optim.zero_grad()
            if opt.DATASET in ['mem' ,'mmhs', 'off']:
                batch_score=compute_score(pred,labels)
            train_score+=batch_score
        print ('Epoch', epoch,'for training loss:',total_loss)
        model.train(False)
        if opt.DATASET=='mem' or opt.DATASET=='off' :
            if opt.DATASET=='off':
                eval_score=0.0
                auc_score=0.0
                test_scores =\
                evaluate_for_offensive(model,test_loader,opt,epoch,loss_fn,
                                       off=True)
                val_scores =\
                evaluate_for_offensive(model,val_loader,opt,epoch,loss_fn,
                                       off=True)
            else:
                eval_score,eval_loss,auc_score =\
                evaluate_for_offensive(model,test_loader,opt,epoch,loss_fn)
            if opt.OFF_EVAL:
                val_scores =\
                evaluate_for_offensive(model,off_val_loader,opt,epoch,loss_fn,
                                       off=True)
                test_scores =\
                evaluate_for_offensive(model,off_test_loader,opt,epoch,loss_fn,
                                       off=True)
        elif  opt.DATASET=='mmhs':
            eval_score,eval_loss,auc_score =\
            evaluate_for_offensive(model,val_loader,opt,epoch,loss_fn)
            test_score,test_loss,test_auc_score =\
            evaluate_for_offensive(model,test_loader,opt,epoch,loss_fn)
        else:
            eval_score,eval_loss,eval_f1,eval_precision,eval_recall =\
            evaluate_for_offensive(model,test_loader,opt,epoch,loss_fn)
        total_loss = total_loss /train_size
        train_score=100 * train_score / train_size
        print (
            'Epoch:',epoch,
            'evaluation score:',eval_score,'eval loss:',eval_loss,
            'train score:',train_score
        )
        logger.write('epoch %d' %(epoch))
        logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss, 
                                                             train_score))
        if opt.DATASET in ['mem' ,'mmhs','off']:
            logger.write('\teval accuracy: %.2f, auc: %.2f ' %\
                         ( eval_score, auc_score))
            if opt.DATASET=='mmhs':
                logger.write('\ttest accuracy: %.2f, test auc: %.2f ' %\
                         ( test_score, test_auc_score))
            if opt.OFF_EVAL or opt.DATASET=='off':
                logger.write('\teval off f1: %.2f, precision: %.2f, recall: %.2f ' %\
                         (val_scores[0],val_scores[1],val_scores[2]))
                logger.write \
                ('\ttest off f1: %.2f, precision: %.2f, recall: %.2f ' %\
                 (test_scores[0],test_scores[1],test_scores[2]))
        else:
            logger.write('\teval accuracy: %.2f ' %\
                         ( eval_score))
            logger.write('\teval f1, precision and recall: %.2f, %.2f,%.2f ' %\
                         ( eval_f1,eval_precision,eval_recall))
        model.train(True)
    if opt.ABLATION==False:
        torch.save(model.state_dict(),
                   os.path.join('/home/ruicao/NLP/datasets/hate-speech/save_models',
                                'latent.pth'))
    return eval_score
    
def evaluate_for_offensive(model,test_loader,opt,epoch,cri,off=False):
    score=0.0
    total_loss=0
    f1=precision=recall=0.0
    acc=0.0
    auc_score=0.0
    total_num=len(test_loader.dataset)
    print ('The length of the loader is:',len(test_loader.dataset))
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            labels=batch['answer'].float().cuda()
            if opt.DATASET in ['mem' ,'mmhs', 'off']:
                labels=labels.view(-1,1)
            feat=batch['feature'].float().cuda()
            if opt.MODEL=='bert' or opt.MODEL=='latent':
                bert_tokens=batch['bert'].long().cuda()
                mask=batch['mask'].long().cuda()
                token_type_ids=batch['type_token'].long().cuda()
                if opt.MODEL=='bert':
                    pred=torch.sigmoid(
                        model(bert_tokens, 
                              token_type_ids=token_type_ids,
                              attention_mask=mask)[0]
                    )
                else:
                    external_tokens=batch['ext'].cuda()
                    ext_mask=batch['ext_mask'].cuda()
                    pred,anno_target,align_target=model(
                        feat,
                        bert_tokens,
                        external_tokens,
                        token_type_ids,
                        mask,
                        ext_mask)
            if opt.MODEL=='latent':
                if opt.DATASET in ['mem' ,'mmhs', 'off']:
                    if off or opt.DATASET=='off':
                        batch_score,b_f1,b_recall,b_pre=compute_other(pred,labels,
                                                                      binary=True)
                        f1+=b_f1
                        precision+=b_pre
                        recall+=b_recall
                    loss_f=cri(pred,labels)
                    batch_score=compute_score(pred,labels)
                    batch_auc=compute_auc_score(pred,labels)
                    auc_score+=batch_auc
                else:
                    loss_f=bce_for_loss(pred,labels)
                    batch_score,b_f1,b_recall,b_pre=compute_other(pred,labels)
                    f1+=b_f1
                    precision+=b_pre
                    recall+=b_recall
                loss_l=bce_for_loss(align_target,anno_target)
                loss=loss_f+opt.LATENT_RATE*loss_l
            else:
                loss=cri(pred,labels)
            loss=cri(pred,labels)
            total_loss+=loss
        
        
        batch_size=feat.shape[0]
        pred=pred.cpu().numpy()
        score+=batch_score
        
        #print (batch_auc)
    score=score*100/len(test_loader.dataset)
    if opt.DATASET in ['mem' ,'mmhs', 'off']:
        auc_score=auc_score*100 /len(test_loader.dataset) 
        print (auc_score)
        if off or opt.DATASET=='off':
            f1=f1*100/len(test_loader.dataset)
            precision=precision*100/len(test_loader.dataset)
            recall=recall*100/len(test_loader.dataset)
            if off:
                return [f1,precision,recall]
            else:
                return [f1,precision,recall], score, total_loss,auc_score
        else:
            return score ,total_loss, auc_score 
    else:
        f1=f1*100/len(test_loader.dataset)
        precision=precision*100/len(test_loader.dataset)
        recall=recall*100/len(test_loader.dataset)
        return score ,total_loss, f1, precision, recall
            
            
            