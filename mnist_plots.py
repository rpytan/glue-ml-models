import matplotlib.pyplot as plt
import numpy as np
import array

# -------------------------------- accuracy as a result of change in batch_size -----------------------------------#
f1 = open("mnist_batch_size.txt", "r")
CELSGD1 = [[], []]
NLLLSGD1 = [[], []]
CELAdam1 = [[], []]
NLLLAdam1 = [[], []]
CELAG1 = [[], []]
NLLLAG1 = [[], []]

for line in f1:
    splitStr = line.split(', ')
    xval = float(splitStr[2])
    yval = float(splitStr[5])
    if splitStr[0] == 'CrossEntropyLoss':
        if splitStr[1] == 'SGD':
            CELSGD1[0].append(xval)
            CELSGD1[1].append(yval)
        elif splitStr[1] == 'Adam':
            CELAdam1[0].append(xval)
            CELAdam1[1].append(yval)
        else:
            CELAG1[0].append(xval)
            CELAG1[1].append(yval)
    else:
        if splitStr[1] == 'SGD':
            NLLLSGD1[0].append(xval)
            NLLLSGD1[1].append(yval)
        elif splitStr[1] == 'Adam':
            NLLLAdam1[0].append(xval)
            NLLLAdam1[1].append(yval)
        else:
            NLLLAG1[0].append(xval)
            NLLLAG1[1].append(yval)
f1.close()

plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.scatter(CELSGD1[0], CELSGD1[1])
plt.scatter(NLLLSGD1[0], NLLLSGD1[1])
plt.scatter(CELAdam1[0], CELAdam1[1])
plt.scatter(NLLLAdam1[0], NLLLAdam1[1])
plt.scatter(CELAG1[0], CELAG1[1])
plt.scatter(NLLLAG1[0], NLLLAG1[1])
plt.legend(['CEL, SGD', 'NLLLoss, SGD', 'CEL, Adam', 'NLLLoss, Adam', 'CEL, AdaGrad', 'NLLLoss, AdaGrad'],
           loc='upper_left')
plt.show()

# ------------------- accuracy as a result of change in number of epochs ------------------------------#
f2 = open("mnist_num_epochs.txt", "r")
CELSGD2 = [[], []]
NLLLSGD2 = [[], []]
CELAdam2 = [[], []]
NLLLAdam2 = [[], []]
CELAG2 = [[], []]
NLLLAG2 = [[], []]

for line in f2:
    splitStr = line.split(', ')
    xval = float(splitStr[3])
    yval = float(splitStr[5])

    if splitStr[0] == 'CrossEntropyLoss':
        if splitStr[1] == 'SGD':
            CELSGD2[0].append(xval)
            CELSGD2[1].append(yval)
        elif splitStr[1] == 'Adam':
            CELAdam2[0].append(xval)
            CELAdam2[1].append(yval)
        else:
            CELAG2[0].append(xval)
            CELAG2[1].append(yval)
    else:
        if splitStr[1] == 'SGD':
            NLLLSGD2[0].append(xval)
            NLLLSGD2[1].append(yval)
        elif splitStr[1] == 'Adam':
            NLLLAdam2[0].append(xval)
            NLLLAdam2[1].append(yval)
        else:
            NLLLAG2[0].append(xval)
            NLLLAG2[1].append(yval)
f2.close()

plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.scatter(CELSGD2[0], CELSGD2[1])
plt.scatter(NLLLSGD2[0], NLLLSGD2[1])
plt.scatter(CELAdam2[0], CELAdam2[1])
plt.scatter(NLLLAdam2[0], NLLLAdam2[1])
plt.scatter(CELAG2[0], CELAG2[1])
plt.scatter(NLLLAG2[0], NLLLAG2[1])
plt.legend(['CEL, SGD', 'NLLLoss, SGD', 'CEL, Adam', 'NLLLoss, Adam', 'CEL, AdaGrad', 'NLLLoss, AdaGrad'],
           loc='upper_left')
plt.show()

# -------------------------- accuracy as a result of change in learning rate ------------------#
lrxval = []
lryval = []

f3 = open("mnist_learning_rate.txt", "r")
for line in f3:
    splitStr = line.split(', ')
    lrxval.append(float(splitStr[0]))
    lryval.append(float(splitStr[1]))

f3.close()

plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.scatter(lrxval, lryval)
plt.show()
