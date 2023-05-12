import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from Bio import SeqIO


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def BLosum_encode(data,blosum):
    Blosum = pd.read_table(blosum,sep='\t',index_col=0)
    # alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    sample_encode = {}
    for i,char in enumerate(data):
        one_char = Blosum[char]
        sample_encode[char+str(i)]=list(one_char)
    return sample_encode

def loadfasta_data(fasta_file):

    seq_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_dict[record.id]=str(record.seq)
    return seq_dict

def all_data_processing(data,blosum):
    #  data encoding
    # AAs = 'ARNDCQEGHILKMFPSTWYV'
    data_encode = {}
    output = {}
    for keyname,j in data.items():
        try:
            seq_length =len(j)
            maaPSSM = np.zeros((seq_length, 20))
            aaPSSM = BLosum_encode(j,blosum)
            for i, aa in enumerate(j):
                # if aa in aaPSSM.keys():
                maaPSSM[i, :] = np.array(aaPSSM[aa+str(i)])
            # output["MaaPSSM"] = maaPSSM
            # data_encode[keyname.split('|')[0].strip()] = maaPSSM.sum(axis=1)/20
            # data_encode[keyname] = maaPSSM.sum(axis=1)/20
            data_encode[keyname] = maaPSSM
        except KeyError:
            pass
    # X = np.array(data_encode)
    return data_encode
