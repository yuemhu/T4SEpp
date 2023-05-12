# -*- coding: utf-8 -*-
#cd /home/hym/T4SEpp/program
#python /home/hym/T4SEpp/program/T4SEpp.py -m T4SEpp -g cuda:1
#/home/hym/software/miniconda3/bin/python /home/hym/T4SEpp/program/T4SEpp.py -m ' . $selected_model . ' -g cuda:1
#/home/hym/software/miniconda3/bin/python /home/hym/T4SEpp/program/T4SEpp.py -m T4SEpp_ESM-1b -g cuda:1
#/home/hym/software/miniconda3/bin/python /home/hym/T4SEpp/program/T4SEpp.py -m T4SEpp_ProtBert -g cuda:1
#/home/hym/software/miniconda3/bin/python /home/hym/T4SEpp/program/T4SEpp.py -m T4SEpp_ProtT5-XL-UniRef50 -g cuda:1

#Author : Yueming Hu
#TIME:2022/11/08

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from Bio import SeqIO

args = sys.argv

def print_run(cmd):
	print(cmd)
	print("")
	os.system(cmd)
def loadfasta_data(fasta_file):
	seq_dict = {}
	C50 = {}
	effector = {}
	output = []
	for record in SeqIO.parse(fasta_file, "fasta"):
		seq = str(record.seq).upper()
		# label = record.description
		description = record.description.split("|")
		if len(description)>3:
			if description[0]=="gi":
				label=description[3].strip()
			else:
				label=description[1].strip()
		else:
			label=description[0].split(" ")[0].strip()
		seq_dict[label]=seq
		output.append(label)
		seq_len = len(seq)
		if seq_len>80:
			C50[label]=seq[seq_len-50:]
			effector[label]=seq[1:seq_len-50]
		else:
			if seq_len>50:
				C50[label]=seq[seq_len-50:]
	f_out = open('input_FL.fa', "w")#创建文件对象
	for key,valueL in seq_dict.items():
		f_out.write('>'+''.join(key) + '\n'+str(valueL) + '\n')
	f_out.close()
	f_out = open('input_C50.fa', "w")#创建文件对象
	for key,valueL in C50.items():
		f_out.write('>'+''.join(key) + '\n'+str(valueL) + '\n')
	f_out.close()
	f_out = open('input_effector.fa', "w")#创建文件对象
	for key,valueL in effector.items():
		f_out.write('>'+''.join(key) + '\n'+str(valueL) + '\n')
	f_out.close()
	return output
def processFL_align(diamond,program):
	#/home/hym/T4SEpp/program/T4SEfinder_training/flT4SE_kfold/flT4SE_K5.dmnd
	print_run(diamond+' blastp --db '+program+'/model/flT4SE/flT4SE.dmnd --query input_FL.fa -e 1e-3 --outfmt 6 --more-sensitive --threads 6 --quiet --id 30 --subject-cover 70 --out input_FL.prot.diamondOut.txt')
	result = open('input_FL.prot.diamondOut.txt')
	reslut_IDlist=[]
	for line in result:
		line0 = line.strip().split("\t")
		if len(line0) == 12:
			Query = line0[0].strip()
			if not reslut_IDlist:
				reslut_IDlist = [Query]
			else:
				reslut_IDlist.append(Query)
	result.close()
	reslut_IDlist = list(set(reslut_IDlist))
	return reslut_IDlist
def openhmmout(filename):
	# filename='/root/data/20220902-T4attention/homology/uniprot_Bacteria_Reference_proteomes_onlyInclude_T4SS/Cag/effectHMM/UP000000429_effector_FAM35.hmmout'
	output = []
	hmmout = open(filename)
	nums=0
	result = hmmout.readlines()
	for k in range(14,len(result)-14):
		line = result[k]
		if line != "\n":
			line0=[i for i in line.split(" ") if i !='']
			if line0[0]=='E-value':
				for j in range(k+2,len(result)-14):
					line = result[j]
					if line != "\n":
						# nums +=1
						line0=[i for i in line.strip().split(' ') if i !='' and i != "\n"]
						if float(line0[0].strip())<1e-4:
							if not output:
								output = [line0[-1].strip()]
							else:
								output.append(line0[-1].strip())
						# print(line0[-1].strip())
					else:
						break
			else:
				k +=1
		else:
			break
	hmmout.close()
	return output
def sighmm(program,hmmsearch):
	# pathfilename = os.getcwd() #获取当前工作目录路径
	pathfilename = program+'/model/sigHMM'
	file_names = os.listdir(pathfilename)
	sigResult = []
	output = []
	for file_name in file_names:
		if file_name.endswith(".hmm"):
			hmmfile = pathfilename + '/' + file_name.strip() #循环地给这些文件名加上它前面的路径，以得到它的具体路径
			print_run(hmmsearch+' -o input_C50_'+file_name.split('.hmm')[0]+'.hmmsigout --noali -E 1e-4 --cpu 6 '+hmmfile+' input_C50.fa')
			output = openhmmout('input_C50_'+file_name.split('.hmm')[0]+'.hmmsigout')
			# print(output)
			sigResult = sigResult+output
	sigResult = list(set(sigResult))
	# print(sigResult)
	return sigResult

def effectorhmm(program,hmmsearch):
	# pathfilename = os.getcwd() #获取当前工作目录路径
	pathfilename = program+'/model/effectorHMM'
	file_names = os.listdir(pathfilename)
	effectorResult = []
	output = []
	for file_name in file_names:
		if file_name.endswith(".hmm"):
			hmmfile = pathfilename + '/' + file_name.strip() #循环地给这些文件名加上它前面的路径，以得到它的具体路径
			print_run(hmmsearch+' -o input_C50_'+file_name.split('.hmm')[0]+'.hmmeffectorout --noali -E 1e-4 --cpu 6 '+hmmfile+' input_effector.fa')
			output = openhmmout('input_C50_'+file_name.split('.hmm')[0]+'.hmmeffectorout')
			effectorResult = effectorResult+output
	effectorResult = list(set(effectorResult))
	# print(effectorResult)
	return effectorResult

def T4attention(T4attentionmodel,cudagpu):
	pathfilename = os.getcwd() #获取当前工作目录路径
	# print_run('/home/hym/software/miniconda3/bin/python /home/hym/T4SEpp/program/extract_pretrained_embedding_folder.py -i '+pathfilename+' -g '+cudagpu+ ' -t '+T4attentionmodel+'>log 2>err')
	deepresult = pd.read_csv('T4attention_'+T4attentionmodel+'_TAprobability.csv')
	deepresult= deepresult[['id','vote']]
	d =  dict([(i,a) for i,a in zip(deepresult['id'], deepresult['vote'])])
	return d
def single4attention(T4attentionmodel,cudagpu):
	pathfilename = os.getcwd() #获取当前工作目录路径
	# print_run('/home/hym/software/miniconda3/bin/python /home/hym/T4SEpp/program/extract_pretrained_embedding_folder.py -i '+pathfilename+' -g '+cudagpu+ ' -t '+T4attentionmodel+'>log 2>err')
	deepresult = pd.read_csv('T4attention_'+T4attentionmodel+'_TAprobability.csv')
	for j in ["k1", "k2", "k3", "k4", "k5"]:
		deepresult[j][deepresult[j]>0.5]=1
		deepresult[j][deepresult[j]<=0.5]=0
	# deepresultmean = deepresult[["k1", "k2", "k3", "k4", "k5"]]
	deepresult["sum"]=deepresult[["k1", "k2", "k3", "k4", "k5"]].sum(axis=1) #取多列数据求和
	deepresult= deepresult[['id','sum']]
	d =  dict([(i,a) for i,a in zip(deepresult['id'], deepresult['sum'])])
	return d
def T4SEpre_bpb(pathfilename,program,pythonbin):
	print_run(pythonbin+' '+program+'/Integbpb_Pos100Aac_50AanFrequency.py -i '+pathfilename+'/input_FL.fa -l 100 -t C-ter -p '+program+'/model/T4SEpre/Pos100AacFrequency -n '+program+'/model/T4SEpre/Neg100AacFrequency -o input_bpb100AacFrequency.data')
	print_run(pythonbin+' '+program+'/5k_model_T4SEpre_SVM_test.py -p '+pathfilename+'/input_bpb100AacFrequency_name.data -k 5 -m '+ program+'/model/T4SEpre/best_model.pkl')
	output = {}
	result = open('input_predict_T4SEpre_bpb.csv')
	for line in result:
		line = line.split(',')
		if len(line)==3 and line[0].strip()!='id':
			output[line[0].strip()]=line[1].strip()
	return output
def listTofile(randomPPIfile,outputfile):
	f_out = open(outputfile, "w")#创建文件对象
	f_out.write('T4SE\thuman\tlabel\n')
	for line in randomPPIfile:
		f_out.write(line[0] + '\t'+line[1] + '\t0\n')
	f_out.close()
def mergeT4SEppsingle(T4merge,sigResult,blastResult,effectorResult,T4SEpre,model,Weight,fastaName): #,kfold
	output = [[] for i in range(len(fastaName))]#T4merge.keys()
	i=0
	T4attentionW,blasthomW,effectorhomW,sighomW,T4SEpreW=Weight[0],Weight[1],Weight[2],Weight[3],Weight[4]
	for key in fastaName:
	# for key,valueL in T4merge.items():
		sighom,blasthom,effectorhom='non-homology','non-homology','non-homology'
		valueL = 0
		T4SEpreValue = 0
		T4SEpreProb = 0
		if key in T4merge:
			valueL = int(T4merge[key])
		valueL1 = 0
		if key in T4SEpre:
			T4SEpreValue = 1 if float(T4SEpre[key]) >= 0.5 else 0
			T4SEpreProb = T4SEpre[key]
		if valueL>=3:
			valueL1 = T4attentionW #0.45
		if key in sigResult:
			valueL1 = valueL1 + max(sighomW,T4SEpreValue*T4SEpreW) #0.25
			sighom = 'homology'
		else:
			valueL1 = valueL1 + T4SEpreValue*T4SEpreW
		if key in blastResult:
			valueL1 = valueL1 + blasthomW #0.1
			blasthom = 'homology'
		if key in effectorResult:
			valueL1 = valueL1 + effectorhomW
			effectorhom = 'homology'
		if valueL1>=0.5:
			valueL1 = min(valueL1,1.0)
			output[i]=[key,model,' '+str(valueL)+' / 5',blasthom,effectorhom,sighom,format(float(T4SEpreProb),".2f"),format(valueL1,".2f"),'T4SE'] #,kfold[key]
		else:
			output[i]=[key,model,' '+str(valueL)+' / 5',blasthom,effectorhom,sighom,format(float(T4SEpreProb),".2f"),format(valueL1,".2f"),'non-T4SE'] #,kfold[key]
		i +=1
	f_out = open('./results/'+model+'_predict_result.csv', 'w')
	for i in range(len(output)):
		line = output[i]
		f_out.write(line[0] + ','+line[2] + ','+line[3] + ','+line[4]+ ','+line[5] + ','+line[6] + ','+line[7]+ ','+line[8]+'\n')
	f_out.close()
def main(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("-i","--inputfile",help="Protein fasta sequence file to be predicted")
	parser.add_argument("-m","--model",default='ProtBert',help="testing model name, please choose ESM-1b, ProtBert or ProtT5-XL-UniRef50")
	parser.add_argument("-b","--pythonbin", default='python',help="python bin folder")
	parser.add_argument("-d","--diamond", default='diamond',help="diamond bin folder")
	parser.add_argument("-s","--hmmsearch", default='hmmsearch',help="hmmsearch bin folder")
	parser.add_argument("-p","--program", default='./program',help="Feature extraction script and model folder")
	parser.add_argument("-g","--cudagpu",default='cuda:0', help="Choose any cuda or cpu for prediction") 
	args = parser.parse_args()
	inputfile = args.inputfile
	model = args.model
	pythonbin = args.pythonbin
	diamond = args.diamond
	program = args.program
	hmmsearch = args.hmmsearch
	cudagpu = args.cudagpu
	blastResult = []
	Bert,UniRef50,ESM1b,T4SEpre={},{},{},{}
	fastaName=[]
	T4merge={}
	if model == "T4SEpp_ESM-1b":
		fastaName=loadfasta_data(inputfile)
		blastResult = processFL_align(diamond,program)
		pathfilename = os.getcwd() #获取当前工作目录路径
		print_run(pythonbin+' '+program+'/extract_pretrained_embedding_folder.py -i '+pathfilename+' -g '+cudagpu+' -t ESM-1b -p '+program)
		sigResult = sighmm(program,hmmsearch)
		effectorResult = effectorhmm(program,hmmsearch)
		ESM1b = single4attention('ESM-1b',cudagpu)
		T4SEpre=T4SEpre_bpb(pathfilename,program,pythonbin)
		w=[0.42,0.44,0.05,0.10,0.10]
		mergeT4SEppsingle(ESM1b,sigResult,blastResult,effectorResult,T4SEpre,model,w,fastaName) #,kfold
		print_run('rm -rf *.yaml input* log err')
	elif model == "T4SEpp_ProtBert":
		fastaName=loadfasta_data(inputfile)
		blastResult = processFL_align(diamond,program)
		pathfilename = os.getcwd() #获取当前工作目录路径
		print_run(pythonbin+' '+program+'/extract_pretrained_embedding_folder.py -i '+pathfilename+' -g '+cudagpu+' -t ProtBert -p '+program)
		sigResult = sighmm(program,hmmsearch)
		T4SEpre=T4SEpre_bpb(pathfilename,program,pythonbin)
		effectorResult = effectorhmm(program,hmmsearch)
		ProtBert = single4attention('ProtBert',cudagpu)
		w=[0.30,0.35,0.08,0.30,0.31]
		mergeT4SEppsingle(ProtBert,sigResult,blastResult,effectorResult,T4SEpre,model,w,fastaName) #,kfold
		print_run('rm -rf *.yaml input* log err')
	elif model == "T4SEpp_ProtT5-XL-UniRef50":
		fastaName=loadfasta_data(inputfile)
		blastResult = processFL_align(diamond,program)
		pathfilename = os.getcwd() #获取当前工作目录路径
		print_run(pythonbin+' '+program+'/extract_pretrained_embedding_folder.py -i '+pathfilename+' -g '+cudagpu+' -t ProtT5-XL-UniRef50 -p '+program)
		sigResult = sighmm(program,hmmsearch)
		T4SEpre=T4SEpre_bpb(pathfilename,program,pythonbin)
		effectorResult = effectorhmm(program,hmmsearch)
		UniRef50 = single4attention('ProtT5-XL-UniRef50',cudagpu)
		w=[0.30,0.38,0.06,0.30,0.31]
		mergeT4SEppsingle(UniRef50,sigResult,blastResult,effectorResult,T4SEpre,model,w,fastaName) #,kfold
		print_run('rm -rf *.yaml input* log err')
	else:
		print('Error: model select err!!!')


if __name__ == '__main__':
	main(args)


