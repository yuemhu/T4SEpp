# -*- coding: utf-8 -*-
#cd /home/hym/data/T4SE_data_fromYIXUE/output/TrainingModel/pretrained_embedding

#Author : Yueming Hu
#TIME:2021/10/30

import argparse
import os
import sys
import gzip
import numpy as np
import torch
from transformers import BertModel, BertTokenizer,AlbertModel, AlbertTokenizer,T5EncoderModel, T5Tokenizer,XLNetModel, XLNetTokenizer
import re
import requests
from tqdm.auto import tqdm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import pathlib
import yaml
from Bio import SeqIO

args = sys.argv
gene_dict = {}
#check device
# def get_device():
# 	return 'cuda:1' if torch.cuda.is_available() else 'cpu'
# # get device 
# device = get_device()
def print_run(cmd):
	print(cmd)
	print("")
	os.system(cmd)

def GetPath(pathfilename):
	#root = os.getcwd() #获取当前工作目录路径
	file_names = os.listdir(pathfilename)
	file_ob_list = {}
	for file_name in file_names:
		if file_name.endswith("_FL.fa"):
			fileob = pathfilename + '/' + file_name.strip() #循环地给这些文件名加上它前面的路径，以得到它的具体路径
			file_ob_list[fileob] = file_name.split('.fa')[0]
			# file_ob_list[fileob] = file_name.strip()
	return file_ob_list
def predictT4SE(experiment_name,test_embeddings,test_fasta,yaml_file,program):
	cofig={
	'experiment_name': 'T4attention_ProtT5-XL-UniRef50',
	'seed': 123,
	'kfold': 5,
	'max_length': 256,
	'num_epochs': 2000,
	'batch_size': 100,
	'num_warmup_steps': 10,
	'optimizer_parameters':{'lr': 1e-4},
	'blosum': '',
	'train_embeddings': '',
	'test_embeddings': '',
	'train_fasta': '',
	'test_fasta': '',
	'model_type': 'T4attention',
	'model_parameters':{
		'dropout': 0.25,
		'kernel_size': 9,
		'output_dim': 2}
	}
	cofig['experiment_name']=experiment_name
	cofig['test_embeddings']=test_embeddings
	cofig['test_fasta']=test_fasta
	with open(yaml_file, 'w', encoding="utf-8") as f:
		yaml.safe_dump(cofig, f)
	print_run("python "+program+"/T4attention_main.py --config "+yaml_file+" --eval_on_test")
	print_run("rm -rf "+test_embeddings)

def openCDhit(filename):
	kind=filename.endswith(".gz")
	output = {}
	if kind:
		protein = gzip.open(filename,'rt', encoding='utf-8')
	else:
		protein = open(filename)
	output = {}
	sequence = ""
	proID = protein.readline()[1:].strip()  # 取第一行
	for line in protein:
		if line[0] == ">":
			output[proID] = sequence
			proID = line[1:].strip()
			sequence = ""
		else:
			sequence +=line.strip()
	output[proID] = sequence
	protein.close()
	return output
def extract_embedding(fasta,tokenizer,model,output,trainmodel,device):
	i = 1
	for key,valueL in fasta.items():
		sequence = ' '.join(list(valueL))
		if trainmodel == 'ProtXLNet':
			sequences = [re.sub(r"[UZOBX]", "<unk>", sequence)]
			# sequences = [re.sub(r"[UZOB]", "X", sequence)]
			# sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
			ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
			input_ids = torch.tensor(ids['input_ids']).to(device)
			attention_mask = torch.tensor(ids['attention_mask']).to(device)
			with torch.no_grad():
				output = model(input_ids=input_ids,attention_mask=attention_mask,mems=None)
				embedding = output.last_hidden_state
				mems = output.mems
			embedding = embedding.cpu()
			seq_len = (attention_mask[0] == 1).sum()
			padded_seq_len = len(attention_mask[0])
			# print(seq_len)
			seq_emd = embedding[0][padded_seq_len-seq_len:padded_seq_len-2]
			# print(seq_emd)
		else:
			sequences = [re.sub(r"[UZOB]", "X", sequence)]
			# sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
			# ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
			ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True,max_length=1280,padding='max_length',truncation='longest_first')
			input_ids = torch.tensor(ids['input_ids']).to(device)
			attention_mask = torch.tensor(ids['attention_mask']).to(device)
			with torch.no_grad():
				embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
			embedding = embedding.cpu()
			seq_len = (attention_mask[0] == 1).sum()
			seq_emd = embedding[0][1:seq_len-1]
		# if i ==1:
		# 	features = seq_emd
		# 	i +=1
		# else:
		# 	features = torch.cat([features,seq_emd],dim=0)
		# print(output+'/'+''.join(key)+'.pt')
		torch.save(seq_emd,output+'/'+''.join(key)+'.pt')
def extract_esm_embedding(key,valueL,alphabet,model,output,trainmodel,device,inputdateset):
	fasta_file = pathlib.Path(''.join(key))
	dataset = FastaBatchedDataset.from_file(fasta_file)
	toks_per_batch = 4096
	repr_layers = [0,33]
	batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
	data_loader = torch.utils.data.DataLoader(
		dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
	print(f"Read {fasta_file} with {len(dataset)} sequences")
	assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
	repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
	include = ["per_tok","mean"]
	return_contacts = "contacts" in include
	with torch.no_grad():
		for batch_idx, (labels, strs, toks) in enumerate(data_loader):
			print(
				f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
			)
			# if torch.cuda.is_available() and not args.nogpu:
			toks = toks.to(device=device, non_blocking=True)

			# The model is trained on truncated sequences and passing longer ones in at
			# infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
			# if args.truncate:
			toks = toks[:, :1022]

			out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

			logits = out["logits"].to(device="cpu")
			representations = {
				layer: t.to(device="cpu") for layer, t in out["representations"].items()
			}
			if return_contacts:
				contacts = out["contacts"].to(device="cpu")

			for i, label in enumerate(labels):
				output_file = inputdateset+'/'+''.join(valueL)+'/'+trainmodel+'/'+label+'.pt'
				result = {"label": label}
				
				# Call clone on tensors to ensure tensors are not views into a larger representation
				# See https://github.com/pytorch/pytorch/issues/1995
				if "per_tok" in include:
					result["representations"] = {
						layer: t[i, 1 : len(strs[i]) + 1].clone()
						for layer, t in representations.items()
					}
				if "mean" in include:
					result["mean_representations"] = {
						layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone()
						for layer, t in representations.items()
					}
				if "bos" in include:
					result["bos_representations"] = {
						layer: t[i, 0].clone() for layer, t in representations.items()
					}
				if return_contacts:
					result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()

				torch.save(
					result,
					output_file,
				)
def main(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("-i","--inputdateset",help="collected Experimentally verified T4S effectors All/cd-hit 30% Training similarity  pathfile")
	parser.add_argument("-t","--trainmodel", help="training model")
	parser.add_argument("-p","--program", default='./program',help="Feature extraction script and model folder")
	parser.add_argument("-g","--cudagpu", help="cudagpu")
	# parser.add_argument("-o","--outputfile",help="embedding feature extraction output pathfile")
	args = parser.parse_args()
	inputdateset = args.inputdateset
	trainmodel = args.trainmodel
	device = args.cudagpu
	program = args.program
	output_dict= {}
	input_dict= {}
	if trainmodel == 'ProtAlbert':
		tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
		model = AlbertModel.from_pretrained("Rostlab/prot_albert")
	elif trainmodel == 'ProtBert':
		tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
		model = BertModel.from_pretrained("Rostlab/prot_bert")
	elif trainmodel == 'ProtT5-XL-BFD':
		tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False )
		model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")
	elif trainmodel == 'ProtBert-BFD':
		tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
		model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
	elif trainmodel == 'ProtT5-XL-UniRef50':
		tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
		model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
	elif trainmodel == 'ProtXLNet':
		tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
		xlnet_men_len = 512
		model = XLNetModel.from_pretrained("Rostlab/prot_xlnet",mem_len=xlnet_men_len)
	elif trainmodel == 'ESM-1b':
		model, alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
	model = model.to(device)
	model = model.eval()
	input_dict = GetPath(inputdateset)
	for key,valueL in input_dict.items():
		output_dict = openCDhit(key)
		outputfileName = inputdateset+'/'+trainmodel+'_'+valueL+'_TAprobability.csv'
		if not os.path.exists(outputfileName):
			print_run('mkdir -p '+inputdateset+'/'+valueL+'/'+trainmodel)
			# pathlib.Path(inputdateset+'/'+valueL+'/'+trainmodel).mkdir(parents=True, exist_ok=True) 
			embedding_folder = inputdateset+'/'+valueL+'/'+trainmodel
			pt_list = []
			pt_list = [pt for pt in os.listdir(embedding_folder) if pt.endswith('.pt')]
			if trainmodel == 'ESM-1b':
				if len(pt_list)>0:
					print('file already exists')
				else:
					print('Processing '+valueL+' file')
					extract_esm_embedding(key,valueL,alphabet,model,embedding_folder,trainmodel,device,inputdateset)
			else:
				if len(pt_list)>0:
					print('file already exists')
				else:
					print('Processing '+valueL+' file')
					extract_embedding(output_dict,tokenizer,model,embedding_folder,trainmodel,device)
			experiment_name = 'T4attention_'+trainmodel
			test_fasta = key
			# yaml_file = inputdateset+'/'+valueL+'.yaml'
			yaml_file = inputdateset+'/'+experiment_name+'.yaml'
			predictT4SE(experiment_name,embedding_folder,test_fasta,yaml_file,program)
			# print_run('mv T4attention_'+trainmodel+'_TAprobability.csv ' +trainmodel+'_'+valueL+'_TAprobability.csv')


if __name__ == '__main__':
	main(args)
