#!/usr/bin/env python
from sympy import *
import numpy as np
import csv
import sys
import os
import math
import random
import matplotlib.pyplot as plt
layer = [8,6,1]
batch = int(sys.argv[1])
epoch = 3000
l_rate = float(sys.argv[2])
w = []
l = []
for i in range(len(layer)-1):
	w.append([[random.uniform(-10,10) for row in range(layer[i])] for col in range(layer[i+1])])
b = []
for i in range(len(layer)-1):
	b.append([0.0 for col in range(layer[i+1])])
def sigmoid(x):
	return 1 / (1 + math.exp(-x))
def one_hot(x):
	a = np.array(x)
	a = np.insert(a,1,0,axis=1)
	a = np.insert(a,2,0,axis=1)
	a = np.insert(a,3,0,axis=1)
	for row in a:
		row[int(row[4])] = 1
	a = np.delete(a,4,axis=1)
	return a
def norm(x):
	sq = 0
	sum =0
	for i in x:
		sum +=float(i[6])
		sq += float(i[6]) *float(i[6])
	sdd = math.sqrt(sq/len(x) - (sum/len(x))*(sum/len(x)))
	for i in x:
		i[6] = (float(i[6]) - sum/len(x))/sdd
	return x
def d_sigmoid(s):
	ss = np.array(s)
	return np.multiply(np.subtract(np.ones(ss.shape) , ss) , ss)
def run(batchs):
	global w
	global b
	E = 0
	all = []
	gtt = []
	for input in batchs:
		gt = int(input[0])
		a = []
		a.append([sigmoid(float(input[i])) for i in range(1,9)])
		for i in range(len(layer)-1):
			a.append(np.add(np.dot(w[i],a[i]), b[i]))
			for front in range(len(a[i+1])):
				a[i+1][front] = sigmoid(a[i+1][front])
		#d = a[len(layer)-1][0] + a[len(layer)-1][1]
		#a[len(layer)-1][0] /= d
		#a[len(layer)-1][1] /= d
		#print(a[len(layer)-1])
		all.append(a)
		gtt.append(gt)
		E -= np.log(a[len(layer)-1][0])*gt + np.log(1 - a[len(layer)-1][0])*(1-gt)
	E/=batch
	#back propagation
	w2 = w
	b2 = b
	for n in range(len(all)):
		l = []
		l.append([all[n][-1][0] - gtt[n]])
		for i in range(len(layer)-1):
			l.append(np.multiply(np.dot(np.array(w[-1-i]).T , l[i]),d_sigmoid(all[n][-2-i])).tolist())
			w2[-1-i] = np.subtract(w2[-1-i] , np.array(np.dot(np.array([l[i]]).T , [all[n][-2-i]]))*l_rate).tolist()
			b2[-1-i] = np.subtract(b2[-1-i] , np.array(l[i]) * l_rate).tolist()
	w = w2
	b = b2

	return E
def testtest(t):
	n = len(t)
	c = 0
	for input in t:
		gt = int(input[0])
		a = []
		a.append([sigmoid(float(input[i])) for i in range(1,9)])
		for i in range(len(layer)-1):
			a.append(np.add(np.dot(w[i],a[i]), b[i]))
			for front in range(len(a[i+1])):
				a[i+1][front] = sigmoid(a[i+1][front])
		if (a[len(layer)-1][0] > 0.5 and gt == 0) or (a[len(layer)-1][0] <= 0.5 and gt == 1):
			c = c + 1
		#print(a[len(layer)-1])
	return float(c/n)

f = open('titanic.csv','r')
titanic = list(csv.reader(f))
data = norm(titanic[1:892])
data = [[float(col) for col in row] for row in data]
data = one_hot(data)
train = data[:800]
test = data[800:891]
print(len(data[0]))
ter = []
err = []
ent = []
for i in range(epoch):
	random.shuffle(train)
	ent.append(run(train[0:int(batch)]))
	err.append(testtest(test))
	ter.append(testtest(train[0:int(batch)]))
testtest(test)
plt.figure()
plt.plot(err)
plt.savefig('test_err_'+str(batch)+'_'+str(l_rate)+'_661_hot.png')
plt.figure()
plt.plot(ent)
plt.savefig('learning_curve_'+str(batch)+'_'+str(l_rate)+'_661_hot.png')
plt.figure()
plt.plot(ter)
plt.savefig('train_err_'+str(batch)+'_'+str(l_rate)+'_661_hot.png')
