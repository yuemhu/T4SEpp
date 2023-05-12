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
#for cuda:3
#python ../5k_model_ESM-1b.py -p /home/hym/software/esm-main/T4SS_c0.3_emb_esm1b -n /home/hym/software/esm-main/nonT4SS_c0.3_emb_esm1b -k 5 >log 2>err &
#python ../5k_model_ESM-1b.py -p /home/hym/software/esm-main/T4SS_c0.3_emb_esm1b -n /home/hym/software/esm-main/nonT4SS_c0.3_emb_esm1b -k 5 -m LightAttention>LAlog 2>LAerr &

#python ../5k_model_ESM-1b.py -p /home/hym/software/esm-main/T4SS_c0.3_emb_esm1b -n /home/hym/software/esm-main/nonT4SS_c0.3_emb_esm1b -k 5 -m LAconvolution >LAClog 2>LACerr &

#python ../5k_model_ESM-1b.py -p /home/hym/software/esm-main/T4SS_c0.3_emb_esm1b -n /home/hym/software/esm-main/nonT4SS_c0.3_emb_esm1b -k 5 -m T4attention >T4Alog 2>T4Aerr &

#AUC 0.7027287, sd = 0.08178071
#TIME:2021/12/14
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler   
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,precision_recall_curve, auc
import sys
import gzip
import math
import pandas as pd

from T4attention.dataset import load_data_T4attention,T4aTraindataSet,collate_fn,Faload_data_T4attention
from T4attention.loss import get_cosine_schedule_with_warmup,FocalLoss,AMSoftmax_FocalLoss
from T4attention.model import T4attention,T4attention_single
import yaml
args = sys.argv

ac_in_dim=200

#check device
def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
# get device 
device = get_device()
# print(f'DEVICE: {device}')
random_state = 632
########k折划分############        
def k_fold(eval_on_test,add_blosum,kfold,max_length,num_epochs,batch_size,num_warmup_steps,optimizer_parameters,train_embeddings,train_fasta,model_type,model_parameters,blosum,experiment_name):
    if model_type == 'LightAttention':
        Train_xy, label,class_weights=load_data_LAattention(train_embeddings,train_fasta,eval_on_test)
    elif model_type == 'T4attention':
        Train_xy,Train_pssm,label,class_weights,T4SE_name,in_dim,T4SE_len=Faload_data_T4attention(train_embeddings,train_fasta,eval_on_test,max_length,blosum,experiment_name)
    else:
        Train_xy, label,class_weights=load_data(train_embeddings,train_fasta,eval_on_test)

    i=0
    stratifiedKFolds = StratifiedKFold(n_splits=kfold, shuffle=True,random_state=random_state)
    # Train_pssm = Train_pssm.astype(float)
    for (trn_idx, val_idx) in stratifiedKFolds.split(Train_pssm, label):
        if model_type == 'T4attention':
            # Train_x,label_x,Train_pssm_x,Train_len_x = Train_xy[trn_idx,:],torch.from_numpy(label[trn_idx]),torch.from_numpy(Train_pssm[trn_idx,:]),T4SE_len[trn_idx]
            # Train_y,label_y,Train_name,Train_pssm_y,Train_len_y =Train_xy[val_idx,:],torch.from_numpy(label[val_idx]),T4SE_name[val_idx],torch.from_numpy(Train_pssm[val_idx,:]),T4SE_len[val_idx]
            Train_x,label_x,Train_pssm_x,Train_len_x = Train_xy[trn_idx],label[trn_idx],Train_pssm[trn_idx],T4SE_len[trn_idx]
            Train_y,label_y,Train_name,Train_pssm_y,Train_len_y =Train_xy[val_idx],label[val_idx],T4SE_name[val_idx],Train_pssm[val_idx],T4SE_len[val_idx]
            data=Train_x,Train_pssm_x,Train_len_x,label_x,Train_y,Train_pssm_y,Train_len_y,label_y
        else:
            Train_x,label_x = torch.from_numpy(Train_xy[trn_idx,:]),torch.from_numpy(label[trn_idx])
        # Train_x,label_x = oversample(Train_xy[trn_idx,:],label[trn_idx])
        # Train_x,label_x = torch.from_numpy(Train_x),torch.from_numpy(label_x)
            Train_y,label_y = torch.from_numpy(Train_xy[val_idx,:]),torch.from_numpy(label[val_idx])
            data=Train_x,label_x,Train_y,label_y
        # Train_x = Train_x.to(device)
        # label_x = label_x.to(device)
        # Train_y = Train_y.to(device)
        # label_y = label_y.to(device)
        '''
        if model_type == 'LightAttention':
            net =  LightAttention(in_dim).to(device)  ### 实例化模型
            model_path="../model/"+experiment_name+"/k"+str(i+1)+"_LAmodel.ckpt"
            torch.save({'data':Train_y,'LAlabel':label_y},
                    "../model/"+experiment_name+"/k"+str(i+1)+"LAvaldata.pt")
        elif model_type == 'T4attention':
            net =  T4attention(max_length,in_dim,model_parameters['output_dim'],model_parameters['dropout'],model_parameters['kernel_size']).to(device)  ### 实例化模型
            model_path="../model/"+experiment_name+"/k"+str(i+1)+"_T4Amodel.ckpt"
            torch.save({'data':Train_y,'label':label_y,'pssm':Train_pssm_y},
                    "../model/"+experiment_name+"/k"+str(i+1)+"T4Avaldata.pt")
            np.save("../model/"+experiment_name+"/k"+str(i+1)+"T4AvaldataName.npy", Train_name)
        else:
            net =  Net(in_dim).to(device)  ### 实例化模型
            model_path="./k"+str(i+1)+"_Netmodel.ckpt"
            torch.save({'data':Train_y,'Netlabel':label_y},
                    "./k"+str(i+1)+"Netvaldata.pt") 
        '''
        ### 每份数据进行训练,体现步骤三####
        if add_blosum:
            model_path="./model/"+experiment_name+'_add_blosum'+"/k"+str(i+1)+"_T4Amodel.ckpt"
            net =  T4attention(max_length,in_dim,model_parameters['output_dim'],model_parameters['dropout'],model_parameters['kernel_size']).to(device)  ### 实例化模型
        else:
            model_path="./model/"+experiment_name+"/k"+str(i+1)+"_T4Amodel.ckpt"
            net =  T4attention_single(max_length,in_dim,model_parameters['output_dim'],model_parameters['dropout'],model_parameters['kernel_size']).to(device)  ### 实例化模型
        i = i + 1
        print('第k'+str(i)+'次traning')
        train(net,add_blosum,max_length, num_epochs, num_warmup_steps,optimizer_parameters,\
                                   class_weights, batch_size,model_path,model_type,i,experiment_name,*data) 


#########训练函数##########
def train(net,add_blosum,max_length, num_epochs,num_warmup_steps, optimizer_parameters,class_weights, batch_size,model_path,model_type,val_num,experiment_name,*data):
    train_ls, test_ls = [], [] ##存储train_loss,test_loss
    # train_features, train_labels = train_features.to(device),train_labels.to(device)
    if model_type == 'T4attention':
        train_emb,train_pssm,Train_len,train_labels, test_emb,test_pssm,Test_len,test_labels = data
        dataset = T4aTraindataSet(train_emb,train_pssm,Train_len,train_labels) 
        val_set = T4aTraindataSet(test_emb,test_pssm,Test_len,test_labels)
        # train_features, train_labels, test_features, test_labels = data
        # dataset = TraindataSet(train_features, train_labels) 
        # val_set = TraindataSet(test_features, test_labels)
    else:
        train_features, train_labels, test_features, test_labels = data
        dataset = TraindataSet(train_features, train_labels) 
        val_set = TraindataSet(test_features, test_labels)


    ### 将数据封装成 Dataloder 对应步骤（2）
    
    #这里使用了Adam优化算法, weight_decay=0
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=float(optimizer_parameters['lr']))
    # optimizer = torch.optim.Adam(params=net.parameters(), lr= 2*1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5) #自衰减调整
    # optimizer = torch.optim.Adam(params=net.parameters(), lr= learning_rate)
    #这里使用了SGD+momentum优化算法
    # optimizer = torch.optim.SGD(params=net.parameters(),lr=1e-4, momentum=0.9)
    # optimizer = torch.optim.Adamax(params=net.parameters(), lr= learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_epochs,num_warmup_steps)
    # loss_func = nn.CrossEntropyLoss(weight=class_weights) ###申明loss函weight=weight_decay
    # loss_func = focal_loss(alpha=class_weights.tolist(), gamma=4, num_classes=2)
    loss_func = FocalLoss(weight=class_weights,reduction='mean', gamma=2)
    # loss_func = AMSoftmax_FocalLoss(weight=class_weights,reduction='mean', gamma=2,device=device)
    loss_func.to(device)
    running_loss = 0.0
    best_acc = 0.0
    min_loss = 1000
    best_mcc = -1.0
    best_prsn=-1.0
    early_stop_cnt=0
    for epoch in range(num_epochs):
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers = 4,pin_memory=True,collate_fn=collate_fn)  #
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,num_workers = 4,pin_memory=True,collate_fn=collate_fn)
        # train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)  #
        # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
        prob_all = []
        label_all = []
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        TP,TN,FN,FP=0,0,0,0
        for feature in train_iter:  ###分批训练 
            if model_type == 'T4attention':
                data_emb,data_pssm,len_data,y = feature
                if add_blosum:
                    data_emb,data_pssm,len_data,y = data_emb.to(device),data_pssm.to(device),len_data.to(device),y.to(device)
                    output  = net(data_emb,data_pssm,len_data)
                else:
                    data_emb,len_data,y = data_emb.to(device),len_data.to(device),y.to(device)
                    output  = net(data_emb,len_data)
            else:
                X, y = feature
                X, y = X.to(device), y.to(device)
                output  = net(X)
            loss = loss_func(output,y)
            optimizer.zero_grad()
            _, train_pred = torch.max(output, dim=1) # get the index of the class with the highest probability
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc += (train_pred.cpu() == y.cpu()).sum().item()
            train_loss += loss.item()
            running_loss +=loss.item()
        if len(val_set) > 0:
            net.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, valfeature in enumerate(val_loader):
                    if model_type == 'T4attention':
                        inputs,inputs1,len_data,labels = valfeature
                        if add_blosum:
                            inputs,inputs1,len_data,labels = inputs.to(device),inputs1.to(device),len_data.to(device),labels.to(device)
                            outputs  = net(inputs,inputs1,len_data)
                        else:
                            inputs,len_data,labels = inputs.to(device),len_data.to(device),labels.to(device)
                            outputs  = net(inputs,len_data)
                    else:
                        inputs, labels = valfeature
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                    loss = loss_func(outputs, labels) 
                    outputs = F.softmax(outputs,dim=1)#计算softmax，即该protein属于各类的概率 byHYM
                    val_prob, val_pred = torch.max(outputs, 1)
                    for j,prob in enumerate(val_prob.cpu().numpy()):
                        if val_pred.cpu().numpy()[j]==1:
                            prob_all.append(prob)
                        else:
                            prob_all.append(1-prob)
                    # prob_all.extend(val_prob.cpu().numpy()) #prob[:,1]返回每一行第二列的数，根据该函数的参数可知，y_score表示的较大标签类的分数，因此就是最大索引对应的那个值，而不是最大索引值
                    label_all.extend(labels.cpu().numpy())

                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += loss.item()
                    TP += ((val_pred.cpu() == 1) & (labels.cpu() == 1)).cpu().sum().item()
                    # TN predict 和 label 同时为0
                    TN += ((val_pred.cpu() == 0) & (labels.cpu() == 0)).cpu().sum().item()
                    # FN                    predict 0 label 1
                    FN += ((val_pred.cpu() == 0) & (labels.cpu() == 1)).cpu().sum().item()
                    # FP                    predict 1 label 0
                    FP += ((val_pred.cpu() == 1) & (labels.cpu() == 0)).cpu().sum().item()
                # acc = (TP+TN)/(TP+TN+FN+FP)
                TP,TN,FN,FP=TP/len(val_set),TN/len(val_set),FN/len(val_set),FP/len(val_set)
                if (TP+FN)!=0:
                    sn = TP/(TP+FN)
                else:
                    sn = 0
                if (TN+FP)!=0:
                    sp = TN/(TN+FP)
                else:
                    sp = 0
                if (TP+FP)!=0:
                    pr =  TP/(TP+FP)
                else:
                    pr = 0
                if sn !=0 and pr !=0:
                    # f1= 2/(1/sn+1/pr)
                    f1=2*TP/(2*TP+FP+FN)
                else:
                    f1=0
                if (TP+FN)*(TP+FP)*(TN+FP)*(TN+FN) !=0:
                    mcc = ((TP*TN)-(FN*FP))/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))
                else:
                    mcc = 0
                fpr, tpr, _ = roc_curve(label_all,prob_all)
                precision, recall, _ = precision_recall_curve(label_all,prob_all)
                print('[{:03d}/{:03d}] Train Acc: {:3.3f} Loss: {:3.3f} | Val Acc: {:3.3f} loss: {:3.3f} sn: {:3.3f} sp: {:3.3f} pr: {:3.3f} f1: {:3.3f} mcc: {:3.3f} AUC: {:3.3f} AUPRC: {:3.3f}'.format(
                    epoch + 1, num_epochs, train_acc/len(dataset), train_loss/len(train_iter), val_acc/len(val_set), val_loss/len(val_loader),sn,sp,pr,f1,mcc,auc(fpr, tpr),auc(recall, precision)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc/len(val_set) > best_acc:
                # if min_loss > val_loss:
                # if best_prsn < sp*sn:
                    best_mcc = mcc
                    min_loss = val_loss
                    early_stop_cnt += 1
                    if sn > 0.82:
                    # if train_acc/len(dataset) >=val_acc/len(val_set):
                        # best_prsn = sp*sn
                        # allprobability=list(map(list, zip(*prob_all)))
                        data=pd.DataFrame(prob_all,columns=['k'+str(val_num)])
                        data.insert(loc=1, column='label', value=label_all)
                        data.to_csv(experiment_name+'_k'+str(val_num)+'.csv',sep=',',index=True, header=True)
                        best_acc = val_acc/len(val_set)
                        torch.save(net.state_dict(), model_path)
                        print('saving model with acc {:.3f} loss {:.3f} mcc {:.3f}'.format(best_acc,min_loss/len(val_loader),best_mcc))
                        early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.3f} Loss: {:3.3f}'.format(
                epoch + 1, num_epochs, train_acc/len(dataset), train_loss/len(train_iter)
            ))
        if early_stop_cnt > 60:
            # Stop training if your model stops improving for 200 epochs.
            break
