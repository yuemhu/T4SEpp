
#python T4attention_main.py --config T4attention_ESM-1b.yaml >T4attention_ESM1-b_T4Alog 2>T4Aerr &
#python T4attention_main.py --config T4attention_ESM-1b.yaml --eval_on_test
#python T4attention_main.py --config T4attention_ProtBert.yaml >T4attention_ProtBert_T4Alog 2>T4Aerr &
#python T4attention_main.py --config T4attention_ProtBert.yaml --eval_on_test
#python T4attention_main.py --config T4attention_ProtT5-XL-UniRef50.yaml >T4attention_ProtT5-XL-UniRef50_T4Alog 2>T4Aerr &
#python T4attention_main.py --config T4attention_ProtT5-XL-UniRef50.yaml --eval_on_test

#python T4attention_main.py --config T4attention_ProtAlbert.yaml >T4attention_ProtAlbert_T4Alog 2>T4Aerr &
#python T4attention_main.py --config T4attention_ProtAlbert.yaml --eval_on_test


#/home/hym/data/smb/GCF_000008525.1
#time python T4attention_main.py --config T4_independent_ESM-1b.yaml --eval_on_test
#time python T4attention_main.py --config T4_independent_ProtBert.yaml --eval_on_test
#time python T4attention_main.py --config T4_independent_ProtAlbert.yaml --eval_on_test
#time python T4attention_main.py --config T4_independent_ProtT5-XL-UniRef50.yaml --eval_on_test
#time python T4attention_main.py --config T4_independent_ProtT5-XL-BFD.yaml --eval_on_test
#time python T4attention_main.py --config T4_independent_ProtBert-BFD.yaml --eval_on_test


#python T4attention_main.py --config T4attention_ESM-1b.yaml --add_blosum >T4attention_ESM1-b_T4Alog_add_blosum 2>T4Aerr &
#python T4attention_main.py --config T4attention_ESM-1b.yaml --add_blosum --eval_on_test

#python T4attention_main.py --config T4attention_ESM-1b.yaml >T4attention_ESM1-b_T4Alog1 2>T4Aerr &
#python T4attention_main.py --config T4attention_ESM-1b.yaml --eval_on_test
import random
import os
import yaml
import argparse
#from T4attention.train import *
from T4attention.test import *
def print_run(cmd):
    print(cmd)
    print("")
    os.system(cmd)
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/T4attention_ESM1-b.yaml')
    parser.add_argument('--eval_on_test',action="store_true", help='train or prediction')
    parser.add_argument('--add_blosum',action="store_true", help='add blosum feature')
    parser.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    parser.add_argument("-g","--cudagpu",default='cuda:0', help="cudagpu")
    args = parser.parse_args()
    if args.config:
        file=open(args.config)
        data = yaml.load(file, Loader=yaml.FullLoader)
    eval_on_test = args.eval_on_test
    add_blosum = args.add_blosum
    cudagpu = args.cudagpu
    kfold = data['kfold']
    max_length = data['max_length']
    num_epochs = data['num_epochs']
    batch_size = data['batch_size']
    num_warmup_steps = data['num_warmup_steps']
    optimizer_parameters = data['optimizer_parameters']
    train_embeddings = data['train_embeddings']
    test_embeddings = data['test_embeddings']
    train_fasta = data['train_fasta']
    test_fasta = data['test_fasta']
    model_type = data['model_type']
    model_parameters = data['model_parameters']
    blosum = data['blosum']
    experiment_name = data['experiment_name']

    if eval_on_test:
        Testing(eval_on_test,add_blosum,kfold,max_length,num_epochs,batch_size,num_warmup_steps,optimizer_parameters,\
            test_embeddings,test_fasta,model_type,model_parameters,blosum,experiment_name,cudagpu)
    else:
        if add_blosum:
            print_run('mkdir -p ./model/'+experiment_name+'_add_blosum')
        else:
            print_run('mkdir -p ./model/'+experiment_name)
        k_fold(eval_on_test,add_blosum,kfold,max_length,num_epochs,batch_size,num_warmup_steps,optimizer_parameters,\
            train_embeddings,train_fasta,model_type,model_parameters,blosum,experiment_name)


if __name__ == '__main__':
    main(args)
