# -*- coding: utf-8 -*-
#Author : Yueming Hu
# cd /home/hym/software/esm-main
# python /home/hym/software/esm-main/extract.py esm1b_t33_650M_UR50S T4SS_c0.3_removeIllegalSequences_Training.fasta T4SS_c0.3_emb_esm1b/ \
#     --repr_layers 0 32 33 --include mean per_tok --truncate
# python /home/hym/software/esm-main/extract.py esm1b_t33_650M_UR50S non-T4SE_c0.3_removeIllegalSequences_TrainingNoHit.fasta nonT4SS_c0.3_emb_esm1b/ \
#     --repr_layers 0 32 33 --include mean per_tok --truncate

# python /home/hym/software/esm-main/extract.py esm1b_t33_650M_UR50S T4SS_c0.3_removeIllegalSequences_Testing.fasta T4SS_c0.3_emb_esm1b_testing/ \
#     --repr_layers 0 32 33 --include mean per_tok --truncate
# python /home/hym/software/esm-main/extract.py esm1b_t33_650M_UR50S non-T4SE_c0.3_removeIllegalSequences_TestingNoHit.fasta nonT4SS_c0.3_emb_esm1b_testing/ \
#     --repr_layers 0 32 33 --include mean per_tok --truncate

#cd /home/hym/data/T4SE_data_fromYIXUE/output/TrainingModel/pretrained_embedding/ESM-1b_CNN
#python ../T4attention/test.py -p /home/hym/software/esm-main/T4SS_c0.3_emb_esm1b_testingNoHit -n /home/hym/software/esm-main/nonT4SS_c0.3_emb_esm1b_testing -k 5 -m T4attention

#AUC 0.7027287, sd = 0.08178071
#TIME:2021/12/14
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve, auc
import sys
import gzip
import math
import pandas as pd
import json

from T4attention.dataset import T4aTestdataSet,testload_data_T4attention,collate_fn_test
from T4attention.model import T4attention,T4attention_single
import yaml
args = sys.argv

#check device
# def get_device():
#     return 'cuda:0' if torch.cuda.is_available() else 'cpu'
# get device 
# device = get_device()
# print(f'DEVICE: {device}')

def Testing(eval_on_test,add_blosum,kfold,max_length,num_epochs,batch_size,num_warmup_steps,optimizer_parameters,test_embeddings,test_fasta,model_type,model_parameters,blosum,experiment_name,device):
    # create testing dataset

    # create model and load weights from checkpoint
    if model_type == 'T4attention':
        test,label,names,in_dim,T4SE_len = testload_data_T4attention(test_embeddings,test_fasta,eval_on_test,max_length,experiment_name)
        test_set=T4aTestdataSet(test,T4SE_len,label)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,num_workers = 4,pin_memory=True,collate_fn=collate_fn_test)
        if add_blosum:
            net = T4attention(max_length,in_dim,model_parameters['output_dim'],model_parameters['dropout'],model_parameters['kernel_size']).to(device)
        else:
            net = T4attention_single(max_length,in_dim,model_parameters['output_dim'],model_parameters['dropout'],model_parameters['kernel_size']).to(device)
    allpredict = []
    allprobability = []
    for i in range(kfold):
        predict = []
        probability = []
        if model_type == 'T4attention':
            if add_blosum:
                model_path="./model/"+experiment_name+'_add_blosum'+"/k"+str(i+1)+"_T4Amodel.ckpt"
            else:
                model_path="./model/"+experiment_name+"/k"+str(i+1)+"_T4Amodel.ckpt"
        else:
            model_path="./k"+str(i+1)+"_LAmodel.ckpt"
        print(model_path)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval() # set the model to evaluation mode
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                if add_blosum:
                    X,X1,X2,y0 = data
                    X,X1,X2 = X.to(device),X1.to(device),X2.to(device)
                    outputs = net(X,X1,X2)
                else:
                    X,X2,y0 = data
                    X,X2 = X.to(device),X2.to(device)
                    outputs = net(X,X2)
                outputs = F.softmax(outputs,dim=1)#softmax
                test_prob, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                for y in test_pred.cpu().numpy():
                    predict.append(y)
                for x,y in enumerate(test_prob.cpu().numpy()):
                    if test_pred.cpu().numpy()[x]==1:
                        probability.append(y)
                    else:
                        probability.append(1-y)
        if i == 0:
            allpredict=predict
            allprobability=probability
        elif i == 1:
            allpredict=[allpredict,predict]
            allprobability=[allprobability,probability]
        else:
            allpredict.append(predict)
            allprobability.append(probability)

    allpredict=list(map(list, zip(*allpredict)))
    data=pd.DataFrame(allpredict,columns=['k1', 'k2', 'k3', 'k4', 'k5'])
    data['vote']=data.mean(axis=1)
    data['vote'][data.vote>=0.6]=1
    data['vote'][data.vote<0.6]=0
    allprobability=list(map(list, zip(*allprobability)))
    data1=pd.DataFrame(allprobability,columns=['k1', 'k2', 'k3', 'k4', 'k5'])
    data1['means']=data1.mean(axis=1)
    data1['vote']=data['vote']
    data1.insert(loc=0, column='id', value=names)
    data1.insert(loc=8, column='label', value=label)
    if model_type == 'T4attention':
        if add_blosum:
            data1.to_csv(experiment_name+'_add_blosum'+'_TAprobability.csv',sep=',',index=False, header=True)
        else:
            data1.to_csv(experiment_name+'_TAprobability.csv',sep=',',index=False, header=True)