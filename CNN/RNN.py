import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import re
import random
L_RATE = 0.0001
BATCH_SIZE = 16
epoch = 1000
dictionary = []
dict_count = []
acc = pd.read_excel('ICLR_accepted.xlsx')
rej = pd.read_excel('ICLR_rejected.xlsx')
classes = ('accepted','rejected')
dataset = [[],[]]
for datas in acc[0]:
	split = re.findall(r'\w+',datas)
	value = []
	for i in range(10):
		if i < len(split):
			if split[i].lower() in dictionary:
				value.append(dictionary.index(split[i].lower())+1)
				dict_count[dictionary.index(split[i].lower())]+=1
			else:
				dictionary.append(split[i].lower())
				dict_count.append(1)
				value.append(len(dictionary))
		else:
			value.append(0)
	dataset[0].append(value)
for datas in rej[0]:
	split = re.findall(r'\w+',datas)
	value = []
	for i in range(10):
		if i < len(split):
			if split[i].lower() in dictionary:
				value.append(dictionary.index(split[i].lower())+1)
				dict_count[dictionary.index(split[i].lower())]+=1
			else:
				dictionary.append(split[i].lower())
				dict_count.append(1)
				value.append(len(dictionary))
		else:
			value.append(0)
	dataset[1].append(value)
dic_count = len(dictionary) #2145 in this case
def catTensor(cat):
	return torch.tensor([cat], dtype=torch.long)
def stringTensor(seq):
	tensor = torch.zeros(10, 1, dic_count+1)
	for i, d in enumerate(seq):
		tensor[i][0][d] = 1
	return tensor
	
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden
	def initHidden(self):
		return torch.zeros(1, self.hidden_size)
n_hidden = 128
rnn = RNN(dic_count+1, n_hidden, 2)
def catout(output):
	top_n, top_i = output.topk(1)
	category_i = top_i[0].item()
	return category_i
criterion = nn.NLLLoss()
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-L_RATE, p.grad.data)

    return output, loss.item()
	
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output
t = [0,0]
t[0] = int(len(dataset[0])*0.8)
t[1] = int(len(dataset[1])*0.8)
def randomtrain():
	cat = random.randint(0,1)
	seq = dataset[cat][random.randint(0,t[cat])]
	cat_tensor = catTensor(cat)
	seq_tensor = stringTensor(seq)
	return cat, seq, cat_tensor, seq_tensor
l_curve = []
acc_tr = []
acc_te = []
for i in range(epoch):
	loss = 0
	for j in range(BATCH_SIZE):
		c,s,ct,st = randomtrain()
		o, l = train(ct, st)
		loss += l
	l_curve.append(loss / BATCH_SIZE)
	tracc = 0
	trtotal = 0
	teacc = 0
	tetotal = 0
	for catag in range(2):
		for index, data in enumerate(dataset[catag]):
			if index > t[catag]:
				tetotal+=1
				if catout(evaluate(stringTensor(data))) == catag:
					teacc+=1
			else:
				trtotal+=1
				if catout(evaluate(stringTensor(data))) == catag:
					tracc+=1
	acc_tr.append(tracc/ trtotal)
	acc_te.append(teacc/ tetotal)
	print(loss / BATCH_SIZE)
plt.figure()
plt.plot(l_curve)
plt.savefig('learning_curve_'+str(BATCH_SIZE)+'_'+str(L_RATE)+'RNN.png')
plt.figure()
plt.plot(acc_te)
plt.savefig('test_accuracy_'+str(BATCH_SIZE)+'_'+str(L_RATE)+'RNN.png')
plt.figure()
plt.plot(acc_tr)
plt.savefig('train_accuracy_'+str(BATCH_SIZE)+'_'+str(L_RATE)+'RNN.png')
			
	


