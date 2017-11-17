import csv

def read_trec_format(tsv_fname):
	questions, answers = [], []
	with open(tsv_fname) as f:
		tsv_reader = csv.reader(f, delimiter='\t')
		for line in tsv_reader:
			questions.append(line[2])
			answers.append(line[3])
	return zip(questions, answers)


if __name__ == '__main__':
	read_trec_format('../data/curated-train.tsv')