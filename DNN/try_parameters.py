import os
batch = 16
for i in range(4):
	rate = 0.01
	for j in range(5):
		rate /= 2
		#os.system("DLHW1_1.py " + str(batch) + " " + str(rate))
		os.system("DLHW1.py " + str(batch) + " " + str(rate))
		rate /= 5
		#os.system("DLHW1_1.py " + str(batch) + " " + str(rate))
		os.system("DLHW1.py " + str(batch) + " " + str(rate))
	batch *= 2