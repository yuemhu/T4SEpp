import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
from torch.utils.data import Dataset
from T4attention.utils import loadfasta_data,all_data_processing
from torch.nn.utils.rnn import pad_sequence


##########定义dataset##########
class TraindataSet(Dataset):
    def __init__(self,train_features,train_labels):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len

class T4aTraindataSet(Dataset):
    def __init__(self,train_emb,train_pssm,train_len,train_labels):
        self.x_data = train_emb
        self.x2_data = train_pssm
        self.train_len = train_len
        if train_labels is not None:
            self.y_data = train_labels
        else:
            self.y_data = None
        self.len = len(self.x_data)
    
    def __getitem__(self,index):
        if self.y_data is not None:
            return self.x_data[index],self.x2_data[index],self.train_len[index],self.y_data[index]
        else:
            return self.x_data[index],self.x2_data[index],self.train_len[index]
    def __len__(self):
        return self.len

class T4aTestdataSet(Dataset):
    def __init__(self,train_emb,train_len,train_labels):
        self.x_data = train_emb
        self.train_len = train_len
        if train_labels is not None:
            self.y_data = train_labels
        else:
            self.y_data = None
        self.len = len(self.x_data)
    
    def __getitem__(self,index):
        if self.y_data is not None:
            return self.x_data[index],self.train_len[index],self.y_data[index]
        else:
            return self.x_data[index],self.train_len[index]
    def __len__(self):
        return self.len

def load_data_T4attention(T4SE_embeddings,T4SE_fasta,eval_on_test,segment_len,blosum,experiment_name):  ###此过程主要是步骤（1）
    T4SEdata = loadfasta_data(T4SE_fasta)
    T4SE_file1 = all_data_processing(T4SEdata,blosum)
    T4SE_dir=os.listdir(T4SE_embeddings)
    # segment_len = 512 #2225
    T4SE_data = []
    T4SE_pssm = []
    T4SE_name = []
    T4SE_mean = []
    T4SE_len = []
    label = []
    for name,pssm in T4SE_file1.items():
        index = name.split('|')
        names = index[0].strip()
        t4se=torch.load(T4SE_embeddings+"/"+name+".pt")
        if experiment_name=="T4attention_ESM-1b":
            temp = t4se['representations'][33]
        else:
            temp = t4se
        lengths_emb = temp.shape[0]
        lengths_pssm = pssm.shape[0]
        T4SE_mean.append(temp.mean(axis=0,keepdim=True))
        # if pssm.shape[0]!=lengths:
            # print(names,'=',pssm.shape,lengths)
        if lengths_emb>segment_len:
            start = lengths_emb - segment_len
            temp = temp[start:,:]
            T4SE_len.append(segment_len)
        else:
            T4SE_len.append(lengths_emb)
        if lengths_pssm>segment_len:
            start = lengths_pssm - segment_len
            T4SE_pssm.append(torch.from_numpy(pssm[start:]))
        else:
            T4SE_pssm.append(torch.from_numpy(pssm))
        T4SE_data.append(temp)
        # T4SE_len.append(temp.shape[0])
        if len(index)>1:
            label.append(int(index[-1].strip()))
        elif len(index)==1:
            label.append(None)
        T4SE_name.append(names)
    Train_emb = pad_sequence(T4SE_data, batch_first=True, padding_value=0) #-1e9
    Train_pssm = pad_sequence(T4SE_pssm, batch_first=True, padding_value=0) #-1e9
    Train_pssm = Train_pssm.type(torch.FloatTensor)
    Train_pssm = np.array(Train_pssm)
    in_dim = T4SE_data[0].shape[1]

    Train_mean=torch.stack(T4SE_mean, dim=0)
    Train_emb=torch.cat([Train_emb,Train_mean], dim=1)
    Train_emb = torch.transpose(Train_emb,2,1)
    T4SE_name = np.array(T4SE_name)
    label = np.array(label)
    #Max(Numberof occurrences in most common class) / (Number of occurrences in rare classes)。即用类别中最大样本数量除以当前类别样本的数量，作为权重系数。
    if eval_on_test:
        return Train_emb,Train_pssm,label,T4SE_name,in_dim,torch.IntTensor(T4SE_len)
        # return Train_emb,Train_pssm,label,T4SE_name
    else:
        
        weights = [1/np.count_nonzero(label==0),1/np.count_nonzero(label==1)]
        class_weights = torch.FloatTensor(weights)
        return Train_emb,Train_pssm,label,class_weights,T4SE_name,in_dim,torch.IntTensor(T4SE_len)



def Faload_data_T4attention(T4SE_embeddings,T4SE_fasta,eval_on_test,segment_len,blosum,experiment_name):  ###此过程主要是步骤（1）
    T4SEdata = loadfasta_data(T4SE_fasta)
    T4SE_file1 = all_data_processing(T4SEdata,blosum)
    T4SE_dir=os.listdir(T4SE_embeddings)
    # segment_len = 512 #2225
    T4SE_data = []
    T4SE_pssm = []
    T4SE_name = []
    # T4SE_mean = []
    T4SE_len = []
    label = []
    for name,pssm in T4SE_file1.items():
        index = name.split('|')
        names = index[0].strip()
        t4se=torch.load(T4SE_embeddings+"/"+name+".pt")
        if experiment_name=="T4attention_ESM-1b":
            temp = t4se['representations'][33]
        else:
            temp = t4se
        lengths_emb = temp.shape[0]
        lengths_pssm = pssm.shape[0]
        # T4SE_mean.append(temp.mean(axis=0,keepdim=True))

        if lengths_emb>segment_len:
            # start = lengths_emb - segment_len
            # temp = temp[start:,:]
            T4SE_len.append(segment_len)
        else:
            T4SE_len.append(lengths_emb)
        T4SE_data.append(temp)
        T4SE_pssm.append(torch.from_numpy(pssm).type(torch.FloatTensor))
        # T4SE_len.append(temp.shape[0])
        if len(index)>1:
            label.append(int(index[-1].strip()))
        elif len(index)==1:
            label.append(None)
        T4SE_name.append(names)
    Train_emb = T4SE_data
    Train_pssm = T4SE_pssm
    # Train_emb = pad_sequence(T4SE_data, batch_first=True, padding_value=0) #-1e9
    # Train_pssm = pad_sequence(T4SE_pssm, batch_first=True, padding_value=0) #-1e9
    # Train_pssm = Train_pssm.type(torch.FloatTensor)
    Train_pssm = np.array(Train_pssm)
    Train_emb = np.array(Train_emb)
    print(T4SE_data)
    in_dim = T4SE_data[0].shape[1]

    # Train_mean=torch.stack(T4SE_mean, dim=0)
    # Train_emb=torch.cat([Train_emb,Train_mean], dim=1)
    # Train_emb = torch.transpose(Train_emb,2,1)
    T4SE_name = np.array(T4SE_name)
    label = np.array(label)
    #Max(Numberof occurrences in most common class) / (Number of occurrences in rare classes)。即用类别中最大样本数量除以当前类别样本的数量，作为权重系数。
    if eval_on_test:
        return Train_emb,Train_pssm,label,T4SE_name,in_dim,torch.IntTensor(T4SE_len)
        # return Train_emb,Train_pssm,label,T4SE_name
    else:
        
        weights = [1/np.count_nonzero(label==0),1/np.count_nonzero(label==1)]
        class_weights = torch.FloatTensor(weights)
        return Train_emb,Train_pssm,label,class_weights,T4SE_name,in_dim,torch.IntTensor(T4SE_len)

def testload_data_T4attention(T4SE_embeddings,T4SE_fasta,eval_on_test,segment_len,experiment_name):  ###此过程主要是步骤（1）
    T4SEdata = loadfasta_data(T4SE_fasta)
    T4SE_dir=os.listdir(T4SE_embeddings)
    # segment_len = 512 #2225
    T4SE_data = []
    T4SE_name = []
    # T4SE_mean = []
    T4SE_len = []
    label = []
    for name in T4SE_dir:
        index = name.split('.pt')
        names = index[0].strip()
        t4se=torch.load(T4SE_embeddings+"/"+name)
        if experiment_name=="T4attention_ESM-1b":
            temp = t4se['representations'][33]
        else:
            temp = t4se
        lengths_emb = temp.shape[0]
        # T4SE_mean.append(temp.mean(axis=0,keepdim=True))

        if lengths_emb>segment_len:
            # start = lengths_emb - segment_len
            # temp = temp[start:,:]
            T4SE_len.append(segment_len)
        else:
            T4SE_len.append(lengths_emb)
        T4SE_data.append(temp)
        # T4SE_len.append(temp.shape[0])
        label.append(None)
        T4SE_name.append(names)
    Train_emb = T4SE_data
    Train_emb = np.array(Train_emb)
    # print(T4SE_data)
    in_dim = T4SE_data[0].shape[1]
    T4SE_name = np.array(T4SE_name)
    label = np.array(label)
    #Max(Numberof occurrences in most common class) / (Number of occurrences in rare classes)。即用类别中最大样本数量除以当前类别样本的数量，作为权重系数。
    if eval_on_test:
        return Train_emb,label,T4SE_name,in_dim,torch.IntTensor(T4SE_len)
        # return Train_emb,Train_pssm,label,T4SE_name
def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    # batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    # batch_size = [len(xi[0]) for xi in batch_data]
    segment_len = [xi[2] for xi in batch_data]
    data_emb = [xi[0] for xi in batch_data]
    data_pssm = [xi[1] for xi in batch_data]
    label = [xi[3] for xi in batch_data]
    # data_emb,data_pssm,segment_len,label=[],[],[],[]
    # for xi in batch_data:
    #     data_emb.append(xi[0])
    #     data_pssm.append(xi[1])
    #     segment_len.append(xi[2])
    #     label.append(xi[3])
    data_mean= []
    data_mean_pssm= []

    for j in data_emb:
        data_mean.append(j.mean(dim=0,keepdim=True))

    for k in data_pssm:
        data_mean_pssm.append(k.mean(dim=0,keepdim=True))

    for i in range(len(data_emb)):
        lengths = data_emb[i].shape[0]
        if lengths>256:
            start = random.randint(0, lengths-256)
            # start = lengths - segment_len[i]
            data_emb[i] = data_emb[i][start:start+256,:]
            data_pssm[i] = data_pssm[i][start:start+256,:]
#             len_data.append(segment_len)
#         else:
#             len_data.append(lengths)
    data_emb_pdd = pad_sequence(data_emb, batch_first=True, padding_value=0)
    data_pssm_pdd = pad_sequence(data_pssm, batch_first=True, padding_value=0)
    if data_emb_pdd.shape[1]<256:
        pdd_emb=torch.zeros(data_emb_pdd.shape[0], 256-data_emb_pdd.shape[1], data_emb_pdd.shape[2])
        pdd_pssm=torch.zeros(data_pssm_pdd.shape[0], 256-data_pssm_pdd.shape[1], data_pssm_pdd.shape[2])
        data_emb_pdd=torch.cat([data_emb_pdd,pdd_emb], dim=1)
        data_pssm_pdd=torch.cat([data_pssm_pdd,pdd_pssm], dim=1)


    data_mean = torch.stack(data_mean, dim=0)
    data_emb_pdd_mean=torch.cat([data_emb_pdd,data_mean], dim=1)
    data_emb_pdd_mean = torch.transpose(data_emb_pdd_mean,2,1)

    data_mean_pssm = torch.stack(data_mean_pssm, dim=0)
    data_pssm_pdd_mean=torch.cat([data_pssm_pdd,data_mean_pssm], dim=1)
    data_pssm_pdd_mean = torch.transpose(data_pssm_pdd_mean,2,1)


    segment_len = torch.tensor(segment_len)

    # data_dim = data_pssm_pdd.shape[1]
    if not any(label):
        return data_emb_pdd_mean.type(torch.FloatTensor),data_pssm_pdd_mean.type(torch.FloatTensor),torch.IntTensor(segment_len),np.array(label)
    else:
        return data_emb_pdd_mean.type(torch.FloatTensor),data_pssm_pdd_mean.type(torch.FloatTensor),torch.IntTensor(segment_len),torch.tensor(label)
    # return data_emb.type(torch.FloatTensor),data_pssm_pdd.type(torch.FloatTensor),torch.IntTensor(segment_len),torch.tensor(label,dtype=torch.float32)
def collate_fn_test(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    # batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    # batch_size = [len(xi[0]) for xi in batch_data]
    segment_len = [xi[1] for xi in batch_data]
    data_emb = [xi[0] for xi in batch_data]
    label = [xi[2] for xi in batch_data]
    # data_emb,data_pssm,segment_len,label=[],[],[],[]
    # for xi in batch_data:
    #     data_emb.append(xi[0])
    #     data_pssm.append(xi[1])
    #     segment_len.append(xi[2])
    #     label.append(xi[3])
    data_mean= []

    for j in data_emb:
        data_mean.append(j.mean(dim=0,keepdim=True))

    for i in range(len(data_emb)):
        lengths = data_emb[i].shape[0]
        if lengths>256:
            start = random.randint(0, lengths-256)
            # start = lengths - segment_len[i]
            data_emb[i] = data_emb[i][start:start+256,:]
#             len_data.append(segment_len)
#         else:
#             len_data.append(lengths)
    data_emb_pdd = pad_sequence(data_emb, batch_first=True, padding_value=0)
    if data_emb_pdd.shape[1]<256:
        pdd_emb=torch.zeros(data_emb_pdd.shape[0], 256-data_emb_pdd.shape[1], data_emb_pdd.shape[2])
        data_emb_pdd=torch.cat([data_emb_pdd,pdd_emb], dim=1)


    data_mean = torch.stack(data_mean, dim=0)
    data_emb_pdd_mean=torch.cat([data_emb_pdd,data_mean], dim=1)
    data_emb_pdd_mean = torch.transpose(data_emb_pdd_mean,2,1)



    segment_len = torch.tensor(segment_len)

    # data_dim = data_pssm_pdd.shape[1]
    if not any(label):
        return data_emb_pdd_mean.type(torch.FloatTensor),torch.IntTensor(segment_len),np.array(label)
    else:
        return data_emb_pdd_mean.type(torch.FloatTensor),torch.IntTensor(segment_len),torch.tensor(label)
    # return data_emb.type(torch.FloatTensor),data_pssm_pdd.type(torch.FloatTensor),torch.IntTensor(segment_len),torch.tensor(label,dtype=torch.float32)