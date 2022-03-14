# -*- coding: utf-8 -*-
# Importing stock libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, PreTrainedModel

# WandB Import the wandb library
import wandb

from torch import cuda

import psutil
import humanize
import os
import GPUtil as GPU

import sys

'''
	Runs to do:
	0 1     Raw queries
	1 5     Overlap last answer
	1 9     Maxsim last answer
	2 10    Maxsim last 2 answers
	3 11    Maxsim last 3 answers
	2 14    Bertsim last 2 answers
	3 15    Bertsim last 3 answers
'''

if len(sys.argv) != 3:
	print("python3 script.py <canard_type> <trec_version>")
	exit(1)
canard_type = int(sys.argv[1])
trec_version = int(sys.argv[2])
print("Canard type:",canard_type)
print("TREC data version",trec_version)

device = 'cuda' if cuda.is_available() else 'cpu'

GPUs = GPU.getGPUs()
# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]

def printm():
	process = psutil.Process(os.getpid())
	print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
	print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()

canard_types = {
	0: "all queries, no answers",
	1: "all queries, last 1 answers",
	2: "all queries, last 2 answers",
	3: "all queries, last 3 answer",
	4: "all queries, all answers"
}

def read_canard(type):
	canard_path = "../data/canard"
	TYPES = ["_all_0","_all_1","_all_2","_all_3","_all_all"]
	df = pd.concat([\
		pd.read_csv(f"{canard_path}/train{TYPES[type]}.csv"),\
		pd.read_csv(f"{canard_path}/dev{TYPES[type]}.csv"),\
		pd.read_csv(f"{canard_path}/test{TYPES[type]}.csv")\
		], ignore_index=True)
	train = df[:-750]
	val = df[-750:]
	return train, val # 40000 examples, in which 750 for val

trec_versions = {
	1: "raw queries",
	2: "rewritten queries",
	3: "full history",
	4: "full history, 2 first sentences",

	5: "overlap last answer",
	6: "overlap last 2 answers",
	7: "overlap last 3 answers",
	8: "overlap all answers",

	9: "maxsim last answer",
	10: "maxsim last 2 answers",
	11: "maxsim last 3 answers",
	12: "maxsim all answers",

	13: "bert sim last answer",
	14: "bert sim last 2 answers",
	15: "bert sim last 3 answers",
	16: "bert sim all answers"
}

def read_trec(year,version):
	assert year==2020 or year==2021
	assert version in range(1,17)
	trec_path = "../data/treccast"
	test = pd.read_csv(f"{trec_path}/{year}/trec{year}_{version}.csv").reset_index()
	return test # about 250 examples

def createCQRdata(canard_type,trec_version):
	train, val = read_canard(canard_type)
	val = pd.concat([\
		val,\
		read_trec(2020,trec_version),\
		read_trec(2021,trec_version)\
		] ,ignore_index=True)
	return train[["Source","Target"]], val[["Source","Target"]]

# smart padding for examples in a batch
def my_collate(batch):
	padded_source_ids = pad_sequence([item['source_ids'] for item in batch], batch_first=True)
	padded_source_mask = pad_sequence([item['source_mask'] for item in batch], batch_first=True)
	padded_target_ids = pad_sequence([item['target_ids'] for item in batch], batch_first=True)
	batch = [{'source_ids':padded_source_ids[i], 'source_mask':padded_source_mask[i],\
			  'target_ids':padded_target_ids[i]} for i in range(len(padded_source_ids))]
	return default_collate(batch)

# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

	def __init__(self, dataframe, tokenizer, source_len, target_len):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.source_len = source_len
		self.target_len = target_len
		self.df_target = self.data.Target
		self.df_source = self.data.Source

	def __len__(self):
		return len(self.df_target)

	def __getitem__(self, index):
		df_source = str(self.df_source[index])
		df_source = ' '.join(df_source.split())

		df_target = str(self.df_target[index])
		df_target = ' '.join(df_target.split())

		source = self.tokenizer.batch_encode_plus([df_source], max_length= self.source_len, padding='longest',return_tensors='pt',truncation=True)
		target = self.tokenizer.batch_encode_plus([df_target], max_length= self.target_len, padding='longest',return_tensors='pt',truncation=True)

		source_ids = source['input_ids'].squeeze()
		source_mask = source['attention_mask'].squeeze()
		target_ids = target['input_ids'].squeeze()
		target_mask = target['attention_mask'].squeeze()

		return {
			'source_ids': source_ids.to(dtype=torch.long), 
			'source_mask': source_mask.to(dtype=torch.long), 
			'target_ids': target_ids.to(dtype=torch.long),
			#'target_ids_y': target_ids.to(dtype=torch.long)
		}

# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we enumerate over the training loader and passed to the defined network 

def train(epoch, tokenizer, model, device, loader, optimizer):
	model.train()
	for _,data in enumerate(loader, 0):
		y = data['target_ids'].to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		labels = y[:, 1:].clone().detach()
		labels[y[:, 1:] == tokenizer.pad_token_id] = -100
		ids = data['source_ids'].to(device, dtype = torch.long)
		mask = data['source_mask'].to(device, dtype = torch.long)

		outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
		loss = outputs[0]
		
		if _%10 == 0:
			wandb.log({"Training Loss": loss.item()})

		#if _%500==0:
		#    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def validate(epoch, tokenizer, model, device, loader):
	model.eval()
	predictions = []
	actuals = []
	with torch.no_grad():
		for _, data in enumerate(loader, 0):
			y = data['target_ids'].to(device, dtype = torch.long)
			ids = data['source_ids'].to(device, dtype = torch.long)
			mask = data['source_mask'].to(device, dtype = torch.long)

			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask, 
				max_length=128, 
				num_beams=3,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True
				)
			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
			target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
			if _%300==0:
				print(f'Completed {_}')

			predictions.extend(preds)
			actuals.extend(target)
	return predictions, actuals

def main(canard_type,trec_version):
	# WandB – Initialize a new run
	wandb.init(project="conversational query reformulation")
	name = f"CQR_{canard_type}_{trec_version}"
	wandb.run.name = name
	wandb.run.notes = canard_types[canard_type] + " | " + trec_versions[trec_version]

	# WandB – Config is a variable that holds and saves hyperparameters and inputs
	# Defining some key variables that will be used later on in the training  
	config = wandb.config          # Initialize config
	config.TRAIN_BATCH_SIZE = 4    # input batch size for training (default: 64)
	config.VALID_BATCH_SIZE = 4    # input batch size for testing (default: 1000)
	config.TRAIN_EPOCHS = 5        # number of epochs to train (default: 10)
	config.VAL_EPOCHS = 1 
	config.LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
	config.SEED = 42               # random seed (default: 42)
	config.MAX_LEN = 512
	config.TARGET_LEN = 128

	# Set random seeds and deterministic pytorch for reproducibility
	torch.manual_seed(config.SEED) # pytorch random seed
	np.random.seed(config.SEED) # numpy random seed
	torch.backends.cudnn.deterministic = True

	# tokenizer for encoding the text
	tokenizer = T5Tokenizer.from_pretrained("t5-base")
	
	# Creation of Dataset and Dataloader
	train_dataset, val_dataset = createCQRdata(canard_type,trec_version)

	print("TRAIN Dataset: {}".format(train_dataset.shape))
	print("TEST Dataset: {}".format(val_dataset.shape))


	# Creating the Training and Validation dataset for further creation of Dataloader
	training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.TARGET_LEN)
	val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.TARGET_LEN)

	# Defining the parameters for creation of dataloaders
	train_params = {
		'batch_size': config.TRAIN_BATCH_SIZE,
		'shuffle': True,
		'num_workers': 0,
		'collate_fn': my_collate
		}

	val_params = {
		'batch_size': config.VALID_BATCH_SIZE,
		'shuffle': False,
		'num_workers': 0,
		'collate_fn': my_collate
		}

	# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
	training_loader = DataLoader(training_set, **train_params)
	val_loader = DataLoader(val_set, **val_params)

	
	# Defining the model. We are using t5-base model and added a Language model layer on top for generation. 
	# Further this model is sent to device (GPU/TPU) for using the hardware.
	#model = T5ForConditionalGeneration.from_pretrained("t5-base",cache_dir="/tmp")
	filename = f"/tmp/{name}_4"
	print(filename)
	model = T5ForConditionalGeneration.from_pretrained(filename,cache_dir="/tmp")
	model = model.to(device)

	# Defining the optimizer that will be used to tune the weights of the network in the training session. 
	optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

	# Log metrics with wandb
	wandb.watch(model, log="all")
	# Training loop
	print('Initiating Fine-Tuning for the model on our dataset')

	#for epoch in range(config.TRAIN_EPOCHS):
	for epoch in range(4,5):
		train(epoch, tokenizer, model, device, training_loader, optimizer)
		# save model
		print(f'Save checkpoint after {epoch+1} epoch')
		model.save_pretrained(save_directory=f"/tmp/{name}_{epoch+1}")

		# Validation loop and saving the resulting file with predictions and acutals in a dataframe.
		# Saving the dataframe as predictions.csv
		print(f'Now generating reformulations on our fine tuned model at epoch {epoch+1} for the validation dataset and saving it in a dataframe')
		for epoch_val in range(config.VAL_EPOCHS):
			predictions, actuals = validate(epoch_val, tokenizer, model, device, val_loader)
			final_df = pd.DataFrame({'Prediction':predictions,'Target':actuals})
			outdir = f'../models/CQR/{name}'
			if not os.path.exists(outdir):
			    os.mkdir(outdir)
			final_df.to_csv(f'{outdir}/predictions_{epoch+1}.csv',sep='\t')
			print('Output Files generated for review')
		
torch.cuda.empty_cache()
main(canard_type,trec_version)