#!/usr/bin/env python
from sympy import *
import numpy as np
import csv
import sys
import os
import math
import random
import matplotlib.pyplot as plt
layer = [6,3,3,2]
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
		gt = [0.0,0.0]
		gt[int(input[0])] = 1.0
		a = []
		a.append([sigmoid(float(input[1])),sigmoid(float(input[2])),sigmoid(float(input[3])),sigmoid(float(input[4])),sigmoid(float(input[5])),sigmoid(float(input[6]))])
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
		E -= np.log(a[len(layer)-1][0])*gt[0] + np.log(1 - a[len(layer)-1][0])*(1-gt[0]) + np.log(a[len(layer)-1][1])*gt[1] + np.log(1 - a[len(layer)-1][1])*(1-gt[1])
	E/=batch
	#back propagation
	w2 = w
	b2 = b
	for n in range(len(all)):
		l = []
		l.append([all[n][len(layer)-1][0] - gtt[n][0], all[n][len(layer)-1][1] - gtt[n][1]])
		for i in range(len(layer)-1):
			l.append(np.multiply(np.dot(np.array(w[len(layer)-2-i]).T , l[i]),d_sigmoid(all[n][len(layer)-2-i])).tolist())
			w2[len(layer)-2-i] = np.subtract(w2[len(layer)-2-i] , np.array(np.dot(np.array([l[i]]).T , [all[n][len(layer)-2-i]]))*l_rate).tolist()
			b2[len(layer)-2-i] = np.subtract(b2[len(layer)-2-i] , np.array(l[i]) * l_rate).tolist()
	w = w2
	b = b2
	#print(w)
	#print(b)
	return E
def testtest(t):
	n = len(t)
	c = 0
	for input in t:
		gt = [0.0,0.0]
		gt[int(input[0])] = 1.0
		a = []
		a.append([sigmoid(float(input[1])),sigmoid(float(input[2])),sigmoid(float(input[3])),sigmoid(float(input[4])),sigmoid(float(input[5])),sigmoid(float(input[6]))])
		for i in range(len(layer)-1):
			a.append(np.add(np.dot(w[i],a[i]), b[i]))
			for front in range(len(a[i+1])):
				a[i+1][front] = sigmoid(a[i+1][front])
		if (a[len(layer)-1][0] > a[len(layer)-1][1] and gt[0] ==0.0) or (a[len(layer)-1][0] <= a[len(layer)-1][1] and gt[1] ==0.0):
			c = c + 1
	return float(c/n)

f = open('titanic.csv','r')
titanic = list(csv.reader(f))
data = norm(titanic[1:892])
data = [[float(col) for col in row] for row in data]
train = data[:800]
test = data[800:891]
coor = np.corrcoef(np.array(data).T)
new1 = [[]]
new2 = [[]]
for i in range(7):
	if coor[0][i] > 0:
		new1[0].append(np.amax(np.array(data)[:,i]))
		new2[0].append(np.amin(np.array(data)[:,i]))
	else:
		new1[0].append(np.amax(np.array(data)[:,i]))
		new2[0].append(np.amin(np.array(data)[:,i]))
ter = []
err = []
ent = []
for i in range(epoch):
	random.shuffle(train)
	ent.append(run(train[0:int(batch)]))
	err.append(testtest(test))
	ter.append(testtest(train[0:int(batch)]))
plt.figure()
plt.plot(err)
plt.savefig('test_err_'+str(batch)+'_'+str(l_rate)+'_6332_new.png')
plt.figure()
plt.plot(ent)
plt.savefig('learning_curve_'+str(batch)+'_'+str(l_rate)+'_6332_new.png')
plt.figure()
plt.plot(ter)
plt.savefig('train_err_'+str(batch)+'_'+str(l_rate)+'_6332_new.png')
