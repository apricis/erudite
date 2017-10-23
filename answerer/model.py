import json
import os
from pprint import pprint

data_folder = "../data/SQuAD"
train_file = "train-v1.1.json"

with open(os.path.join(data_folder, train_file)) as f:
	data = json.load(f)["data"]
	for p in data[0]['paragraphs']:
		context, qas = p['context'], p['qas']
		for qa in qas:
			print(qa['question'])
			for a in qa['answers']:
				print(a['text'])
				print(context[a['answer_start']:a['answer_start']+50])
			print()