# -*- coding: utf-8 -*-
#cd /mnt/c/Users/Saige/Desktop/paper/T4SE_tools/T4SE_datasets
#python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_Training.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o T4SS_c0.3_removeIllegalSequences_Trainingbpb100AacFrequency.data
#python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_Training.fasta -l 50 -t N-ter -p Pos50AanFrequency -n Neg50AanFrequency -o T4SS_c0.3_removeIllegalSequences_Trainingbpb50AanFrequency.data
#python Integbpb_Pos100Aac_50AanFrequency.py -i non-T4SE_c0.3_removeIllegalSequences_TrainingNoHit.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o non-T4SE_c0.3_removeIllegalSequences_Trainingbpb100AacFrequency.data
#python Integbpb_Pos100Aac_50AanFrequency.py -i non-T4SE_c0.3_removeIllegalSequences_TrainingNoHit.fasta -l 50 -t N-ter -p Pos50AanFrequency -n Neg50AanFrequency -o non-T4SE_c0.3_removeIllegalSequences_Trainingbpb50AanFrequency.data

#cd C:\Users\Saige\Desktop\T4SE_data_fromYIXUE\bpbAac_by_HYM
# python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_Testing.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o T4SS_c0.3_removeIllegalSequences_Testingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_Testing.fasta -l 50 -t N-ter -p Pos50AanFrequency -n Neg50AanFrequency -o T4SS_c0.3_removeIllegalSequences_Testingbpb50AanFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i non-T4SE_c0.3_removeIllegalSequences_TestingNoHit.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o non-T4SE_c0.3_removeIllegalSequences_Testingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i non-T4SE_c0.3_removeIllegalSequences_TestingNoHit.fasta -l 50 -t N-ter -p Pos50AanFrequency -n Neg50AanFrequency -o non-T4SE_c0.3_removeIllegalSequences_Testingbpb50AanFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_TestingNoHit.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o T4SS_c0.3_removeIllegalSequences_Testingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_TestingNoHit.fasta -l 50 -t N-ter -p Pos50AanFrequency -n Neg50AanFrequency -o T4SS_c0.3_removeIllegalSequences_Testingbpb50AanFrequency.data

# python Integbpb_Pos100Aac_50AanFrequency.py -i non-T4SE_c0.3_removeIllegalSequences_TrainingNoHit.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o non-T4SE_c0.3_removeIllegalSequences_Trainingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i non-T4SE_c0.3_removeIllegalSequences_TestingNoHit.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o non-T4SE_c0.3_removeIllegalSequences_Testingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_TestingNoHit.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o T4SS_c0.3_removeIllegalSequences_Testingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i T4SS_c0.3_removeIllegalSequences_TrainingNoHit_revise.fasta -l 100 -t C-ter -p Pos100AacFrequency -n Neg100AacFrequency -o T4SS_c0.3_removeIllegalSequences_Trainingbpb100AacFrequency.data

#cd /root/data/20220902-T4attention/Retrain_T4SEpre
# python Integbpb_Pos100Aac_50AanFrequency.py -i ../T4attention_neg_traing_C100_c0.3.fa -l 100 -t C-ter -p ../Pos100AacFrequency -n ../Neg100AacFrequency -o T4attention_neg_c0.3_Trainingbpb100AacFrequency.data
# python Integbpb_Pos100Aac_50AanFrequency.py -i ../homology/T4SEfinder_pos_training_C100_c0.3.fa -l 100 -t C-ter -p ../Pos100AacFrequency -n ../Neg100AacFrequency -o T4SEfinder_pos_c0.3_removeIllegalSequences_Trainingbpb100AacFrequency.data

# python Integbpb_Pos100Aac_50AanFrequency.py -i T4_independent_for_all_pos_removeTraining_c0.6-remove4YK1_addneg.fa -l 100 -t C-ter -p ../Pos100AacFrequency -n ../Neg100AacFrequency -o T4_independent_bpb100AacFrequency.data 

import argparse
import os
import sys
import gzip
import pandas as pd
import numpy as np


args = sys.argv
gene_dict = {}
def print_run(cmd):
	print(cmd)
	print("")
	os.system(cmd)

def GetPath(pathfilename):
	#root = os.getcwd() #获取当前工作目录路径
	file_names = os.listdir(pathfilename)
	file_ob_list = {}
	for file_name in file_names:
		if file_name.endswith(".gz"):
			fileob = pathfilename + '/' + file_name.strip() #循环地给这些文件名加上它前面的路径，以得到它的具体路径
			#file_ob_list[fileob] = file_name.split('.')[0]
			file_ob_list[fileob] = file_name.strip()
	return file_ob_list
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False
def openCDhit(filename):
	kind=filename.endswith(".gz")
	output = {}
	if kind:
		protein = gzip.open(filename,'rt', encoding='utf-8')
	else:
		protein = open(filename)
	#first_line = eQTL.readline()  # 取第一行
	#first_line = first_line.split('\t')
	#key = 0
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
def fastaDictTofile(output_dict,outputfile):
	f_out = open(outputfile, "w")#创建文件对象
	for key,valueL in output_dict.items():
		f_out.write('>'+''.join(key) + '\n'+''.join(valueL) + '\n')
	f_out.close()
def DictTofile(output_dict,outputfile):
	f_out = open(outputfile, "w")#创建文件对象
	for key,valueL in output_dict.items():
		f_out.write(''.join(key) + '\t'+str(valueL) + '\n')
	f_out.close()
def FeatureTofile(output_dict,outputfile):
	# f_out = open(outputfile, "w")#创建文件对象
	# for key,valueL in output_dict.items():
	# 	length = len(valueL)
	# 	for i in range(length):
	# 		if i == length - 1 :
	# 			f_out.write(str(valueL[i])+'\n')
	# 		else:
	# 			f_out.write(str(valueL[i])+ ',')
	# f_out.close()
	f_out = open(outputfile.split(".data")[0]+"_name.data", "w")#创建文件对象
	for key,valueL in output_dict.items():
		length = len(valueL)
		f_out.write(key+',')
		for i in range(length):
			if i == length - 1 :
				f_out.write(str(valueL[i])+'\n')
			else:
				f_out.write(str(valueL[i])+ ',')
		# f_out.write('\n')
	f_out.close()
def Frequency(outputNter,outputCter):
	Nter = pd.DataFrame(outputNter)
	Cter = pd.DataFrame(outputCter)
	outputNter = {}
	outputCter = {}
	for i in range(len(Nter)):
		Freq=Nter.iloc[i].value_counts(dropna=False, normalize=True)
		for j in range(len(Freq.index)):
			keys=Freq.index[j]+str(i+1)
			outputNter[keys]=Freq[j]
	for i in range(len(Cter)):
		Freq=Cter.iloc[i].value_counts(dropna=False, normalize=True)
		for j in range(len(Freq.index)):
			keys=Freq.index[j]+str(i+1)
			outputCter[keys]=Freq[j]
	return outputNter,outputCter
def ReadFrequencyMatrix(PosAaFrequencyinput):
	PosAaFrequency = open(PosAaFrequencyinput)
	PosAaFrequency_dict = {}
	for line in PosAaFrequency:
		line = line.split('\t')
		if len(line) == 2:
			keys = line[0].strip()
			PosAaFrequency_dict[keys]=line[1].strip()
	return PosAaFrequency_dict
def truncatPro(output_dict,length,Terminal):
	outputTer = {}
	for key,valueL in output_dict.items():
		valueL = ''.join(valueL)
		seq_len = len(valueL)
		if Terminal == 'N-ter':
			if seq_len>int(length):
				outputTer[key]=list(valueL[1:int(length)+1]) #value转换成list
			else:
				outputTer[key]=list(valueL[1:])
		else:
			if seq_len>int(length):
				outputTer[key]=list(valueL[seq_len-int(length):]) #value转换成list
			else:
				outputTer[key]=list(valueL[1:]) #value转换成list
	return outputTer
def extractPosFeature(input_dict,PosAaFrequency_dict,length):
	PosFeature = {}
	for key,valueL in input_dict.items():
		seqLen = len(valueL)
		PosValue = []
		for i in  range(length):
			if i <seqLen:
				j=valueL[i]+str(i+1)
				if j in PosAaFrequency_dict.keys():
					PosValue.append(PosAaFrequency_dict[j])
				else:
					PosValue.append('0')
			else:
				PosValue.append('0')
		PosFeature[key]=PosValue
	return PosFeature
def extractNegFeature(input_dict,length):
	NegFeature = {}
	aa = 'ACDEFGHIKLMNPQRSTVWY'
	aaSet = list(aa)
	for key,valueL in input_dict.items():
		NegValue = []
		for i in aaSet:
			NegValue.append(valueL.count(i)/int(length))
		NegFeature[key]=NegValue
	return NegFeature
def mergeFeature(NegFeature,PosFeature):
	Features = {}
	for key,valueL in NegFeature.items():
		# print(valueL)
		Features[key] = valueL+PosFeature[key]
	return Features
def main(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("-i","--inputdateset",help="collected Experimentally verified T4S effectors All/cd-hit 30% Training similarity  pathfile")
	parser.add_argument("-f","--formatchange", default=False,help="collected Experimentally verified T4S effectors cd-hit 30% similarity format change")
	parser.add_argument("-l","--length",type=int, default=100,help="Truncated C-terminal 100Aa number exclude length")
	parser.add_argument("-p","--frequencyMatrixPos",help="C-terminal/N-terminal Postivate frequency Matrix")
	parser.add_argument("-n","--frequencyMatrixNeg",help="C-terminal/N-terminal Negative frequency Matrix")
	parser.add_argument("-t","--terminal",type=str, default="C-ter",help="C-terminal/N-terminal, default = C-ter")
	parser.add_argument("-o","--outputfileTer",help="xxAa C-terminal/N-terminal position-specific features pathfile")
	args = parser.parse_args()
	frequencyMatrixPos=args.frequencyMatrixPos
	frequencyMatrixNeg=args.frequencyMatrixNeg
	inputfile = args.inputdateset
	formatchange = args.formatchange
	length = args.length
	Terminal = args.terminal
	outputfileTer = args.outputfileTer
	output_dict = {}
	outputTer = {}
	PosFeatureTer = {}
	frequencyMatrixPos_dict = {}
	NegFeatureTer = {}
	Ter = {}
	frequencyMatrixNeg_dict = {}
	output_dict = openCDhit(inputfile)
	if formatchange:
		DictTofile(output_dict,formatchange)
	outputTer = truncatPro(output_dict,length,Terminal)
	frequencyMatrixPos_dict = ReadFrequencyMatrix(frequencyMatrixPos)
	frequencyMatrixNeg_dict = ReadFrequencyMatrix(frequencyMatrixNeg)
	PosFeatureTer = extractPosFeature(outputTer,frequencyMatrixPos_dict,length)
	NegFeatureTer = extractPosFeature(outputTer,frequencyMatrixNeg_dict,length)
	# NegFeatureTer = extractNegFeature(outputTer,length)
	Ter=mergeFeature(NegFeatureTer,PosFeatureTer)
	FeatureTofile(Ter,outputfileTer)

if __name__ == '__main__':
	main(args)
