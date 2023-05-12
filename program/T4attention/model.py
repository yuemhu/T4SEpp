import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

#check device
def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
# get device 
device = get_device()
# print(f'DEVICE: {device}')

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
def Add_Norm(x1,x2):
    x = torch.cat([x1,x2], dim=-1)
    layer_norm = nn.LayerNorm(x.size()[1:],device=x.device)
    out = layer_norm(x)
    return out

class T4attention(nn.Module):
    def __init__(self,segment_len=256, embeddings_dim=1280, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(T4attention, self).__init__()

        self.feature_conv = nn.Sequential(nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,padding=kernel_size // 2),
                                            nn.Dropout(conv_dropout))
        self.attention_conv = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,padding=kernel_size // 2)
        self.masked_softmax = masked_softmax

        self.softmax = nn.Softmax(dim=-1)

        self.mean_convolution =  nn.Sequential(nn.Conv1d(1, 1, kernel_size, stride=1,padding=kernel_size // 2),
                                                nn.Dropout(conv_dropout))
        self.mean_attention = nn.Conv1d(1, 1, kernel_size, stride=1,padding=kernel_size // 2)

        self.resnet_block1 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=4,kernel_size=3,padding=1),
                   nn.BatchNorm1d(4), nn.ReLU(),
                   nn.Conv1d(in_channels=4,out_channels=4,kernel_size=3,padding=1))

        self.feature_pssm = nn.Sequential(nn.Conv1d(20, 20, kernel_size, stride=1,padding=kernel_size // 2),
                                            nn.Dropout(conv_dropout))
        self.attention_pssm = nn.Conv1d(20, 20, kernel_size, stride=1,padding=kernel_size // 2)

        self.mean_conv_pssm =  nn.Sequential(nn.Conv1d(1, 1, kernel_size, stride=1,padding=kernel_size // 2),
                                                nn.Dropout(conv_dropout))
        self.mean_attention_pssm = nn.Conv1d(1, 1, kernel_size, stride=1,padding=kernel_size // 2)

        self.resnet_block2 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
                   nn.BatchNorm1d(16), nn.ReLU(),
                   nn.Conv1d(in_channels=16,out_channels=16,kernel_size=3,padding=1))


        self.maxpool = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(conv_dropout)
        self.dropout2 = nn.Dropout(conv_dropout)
        self.Add_Norm = Add_Norm
        #+segment_len
        self.mlp = nn.Sequential(
            nn.Linear(((4*embeddings_dim+16*20)//2), embeddings_dim//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(embeddings_dim//2),
            nn.Linear(embeddings_dim//2, output_dim)
        )


    def forward(self, emb,pssm,seq_len):
        """
        :param x: torch.Tensor [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
        :return: [batch_size,output_dim] tensor with logits
        """
        # print(x.shape)
        emb1,emb2 = emb[:,:,:-1],emb[:,:,-1]

        intermediate_state = self.feature_conv(emb1)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_conv(emb1)  # [batch_size, embeddings_dim, sequence_length]
        # take the weighted sum according to the attention scores
        emb_attention = torch.sum(intermediate_state * self.masked_softmax(attention,seq_len),dim=-1)  # [batchsize, embeddings_dim]

        emb2 = emb2.unsqueeze(1)
        intermediate_state = self.mean_convolution(emb2)  # [batch_size, embeddings_dim, sequence_length]
        intermediate_state = self.dropout2(intermediate_state)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.mean_attention(emb2)  # [batch_size, embeddings_dim, sequence_length]
        emb2 = intermediate_state * self.softmax(attention)

        emb_attention = emb_attention.unsqueeze(1)+emb2
        emb_attention = self.maxpool(F.relu(self.resnet_block1(emb_attention)+emb_attention))
        emb_attention = emb_attention.reshape(emb_attention.size(0),-1)

        pssm1,pssm2 = pssm[:,:,:-1],pssm[:,:,-1]

        intermediate_state = self.feature_pssm(pssm1)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_pssm(pssm1)  # [batch_size, embeddings_dim, sequence_length]
        # take the weighted sum according to the attention scores
        pssm_attention = torch.sum(intermediate_state * self.masked_softmax(attention,seq_len),dim=-1)  # [batchsize, embeddings_dim]

        pssm2 = pssm2.unsqueeze(1)
        intermediate_state = self.mean_convolution(pssm2)  # [batch_size, embeddings_dim, sequence_length]
        intermediate_state = self.dropout2(intermediate_state)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.mean_attention(pssm2)  # [batch_size, embeddings_dim, sequence_length]
        pssm2 = intermediate_state * self.softmax(attention)

        pssm_attention = pssm_attention.unsqueeze(1)+pssm2
        pssm_attention = self.maxpool(F.relu(self.resnet_block2(pssm_attention)+pssm_attention))
        pssm_attention = pssm_attention.reshape(pssm_attention.size(0),-1)



        x = self.Add_Norm(emb_attention,pssm_attention)
        x = self.mlp(x)  # [batchsize, 32]
        return x # [batchsize, output_dim]

class T4attention_single(nn.Module):
    def __init__(self,segment_len=256, embeddings_dim=1280, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(T4attention_single, self).__init__()

        self.feature_conv = nn.Sequential(nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,padding=kernel_size // 2),
                                            nn.Dropout(conv_dropout))
        self.attention_conv = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,padding=kernel_size // 2)
        self.masked_softmax = masked_softmax

        self.softmax = nn.Softmax(dim=-1)

        self.mean_convolution =  nn.Sequential(nn.Conv1d(1, 1, kernel_size, stride=1,padding=kernel_size // 2),
                                                nn.Dropout(conv_dropout))
        self.mean_attention = nn.Conv1d(1, 1, kernel_size, stride=1,padding=kernel_size // 2)

        self.resnet_block1 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=4,kernel_size=3,padding=1),
                   nn.BatchNorm1d(4), nn.ReLU(),
                   nn.Conv1d(in_channels=4,out_channels=4,kernel_size=3,padding=1))


        self.maxpool = nn.MaxPool1d(2)

        self.dropout1 = nn.Dropout(conv_dropout)
        self.dropout2 = nn.Dropout(conv_dropout)
        self.Add_Norm = Add_Norm
        #+segment_len
        self.mlp = nn.Sequential(
            nn.Linear((4*embeddings_dim)//2, embeddings_dim//2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(embeddings_dim//2),
            nn.Linear(embeddings_dim//2, output_dim)
        )


    def forward(self,emb,seq_len):
        """
        :param x: torch.Tensor [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
        :return: [batch_size,output_dim] tensor with logits
        """
        # print(x.shape)
        emb1,emb2 = emb[:,:,:-1],emb[:,:,-1]

        intermediate_state = self.feature_conv(emb1)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_conv(emb1)  # [batch_size, embeddings_dim, sequence_length]
        # take the weighted sum according to the attention scores
        emb_attention = torch.sum(intermediate_state * self.masked_softmax(attention,seq_len),dim=-1)  # [batchsize, embeddings_dim]

        emb2 = emb2.unsqueeze(1)
        intermediate_state = self.mean_convolution(emb2)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.mean_attention(emb2)  # [batch_size, embeddings_dim, sequence_length]
        emb2 = intermediate_state * self.softmax(attention)

        emb_attention = emb_attention.unsqueeze(1)+emb2
        layer_norm = nn.LayerNorm(emb_attention.size()[1:],device=emb_attention.device)
        emb_attention = layer_norm(emb_attention)

        emb_attention = self.maxpool(F.relu(self.resnet_block1(emb_attention)+emb_attention))
        emb_attention = emb_attention.reshape(emb_attention.size(0),-1)

        x = self.mlp(emb_attention)  # [batchsize, 32]
        return x # [batchsize, output_dim]