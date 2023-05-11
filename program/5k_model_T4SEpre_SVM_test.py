# -*- coding: utf-8 -*-
#Author : Yueming Hu

#cd /root/data/20220902-T4attention/Retrain_T4SEpre
#python 5k_model_T4SEpre_SVM_test.py -p T4_independent_bpb100AacFrequency_name.data -k 5
#cd /root/data/20220902-T4attention/Retrain_T4SEpre/PS
#python ../5k_model_T4SEpre_SVM_test.py -p ../T4_independent_ps100AacFrequency_name.data -k 5


from sklearn import datasets
from sklearn import svm
import joblib
import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
import gzip
import math
args = sys.argv

########k折划分############        
def load_data(T4SE):  ###此过程主要是步骤（1）
    T4SE_file1 =np.loadtxt(T4SE,delimiter=',',dtype=str)
    T4SE_file =T4SE_file1[:,1:].astype(np.float32)
    name=T4SE_file1[:,0].tolist()
    return T4SE_file,name


def test(k, T4SE,model_path):
    test,names = load_data(T4SE)
    allpredict = []
    allprobability = []
    predict = []
    probability = []
    model = joblib.load(model_path)
    test_prob = model.predict_proba(test)[:, 1]
    for y in test_prob:
        probability.append(y)
        label = 1 if y >= 0.5 else 0
        predict.append(label)
    allpredict=predict
    allprobability=probability
    data=pd.DataFrame(allpredict,columns=['predict'])
    data1=pd.DataFrame(allprobability,columns=['probability'])
    data1['predict']=data['predict']
    data1.insert(loc=0, column='id', value=names)
    data1.to_csv(T4SE.split('/')[-1].split('_')[0].split('.')[0]+'_predict_T4SEpre_bpb.csv',sep=',',index=False, header=True)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--T4SE', required=True)
    parser.add_argument('-k', '--kfold', default=5)
    parser.add_argument('-m', '--model', default="/home/hym/T4SEpp/program/model/T4SEpre/best_model.pkl")

    args = parser.parse_args()
    T4SE = args.T4SE
    kfold = args.kfold
    model = args.model
    
    test(int(kfold),T4SE,model)



if __name__ == '__main__':
    main(args)