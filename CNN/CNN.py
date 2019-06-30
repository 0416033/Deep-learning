import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
EPOCH = 100
BATCH_SIZE = 16
L_RATE = 0.001
def run(): 
	
	torch.multiprocessing.freeze_support()
	traindata = torchvision.datasets.ImageFolder('animal-10/train/',transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
	trainloader = torch.utils.data.DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)									  
	testdata = torchvision.datasets.ImageFolder('animal-10/val/',transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
	testloader = torch.utils.data.DataLoader(testdata, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)		
	print(len(traindata))
	print(len(trainloader))
	print(traindata.class_to_idx)
	# show images
	# print labels
	
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)
			self.pool = nn.MaxPool2d(2, 2)
			self.conv2 = nn.Conv2d(6, 16, 5 ,1, 2)
			self.fc1 = nn.Linear(16*64*64, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)

		def forward(self, x):
			x = self.pool(F.relu(self.conv1(x)))
			x = self.pool(F.relu(self.conv2(x)))
			x = x.view(-1, 16*64*64)
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x
	net = Net()
	l_curve = []
	acc_tr = []
	acc_te = []
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=L_RATE, momentum=0.9)
		
	for epoch in range(EPOCH):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 625 == 624:    # print every 625 mini-batches
				print('[%d, %5d] loss: %.3f' %
					  (epoch + 1, i + 1, running_loss / 625))
				l_curve.append(running_loss / 625)
				correct = 0
				total = 0
				with torch.no_grad():
					for data in testloader:
						images, labels = data
						outputs = net(images)
						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels).sum().item()
				acc_te.append(correct / total)
				correct = 0
				total = 0
				with torch.no_grad():
					for data in trainloader:
						images, labels = data
						outputs = net(images)
						_, predicted = torch.max(outputs.data, 1)
						total += labels.size(0)
						correct += (predicted == labels).sum().item()
				acc_tr.append(correct / total)
				running_loss = 0.0
				
	print('Finished Training')
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(BATCH_SIZE):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1
				
	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (
			i, 100 * class_correct[i] / class_total[i]))
	plt.figure()
	plt.plot(l_curve)
	plt.savefig('learning_curve_'+str(BATCH_SIZE)+'_'+str(L_RATE)+'.png')
	plt.figure()
	plt.plot(acc_te)
	plt.savefig('test_accuracy_'+str(BATCH_SIZE)+'_'+str(L_RATE)+'.png')
	plt.figure()
	plt.plot(acc_tr)
	plt.savefig('train_accuracy_'+str(BATCH_SIZE)+'_'+str(L_RATE)+'.png')

	
if __name__ == '__main__':
    run()

										  

