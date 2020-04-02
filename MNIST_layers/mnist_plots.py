import matplotlib.pyplot as plt
import numpy as np
import array

# -------------------------------- accuracy as a result of change in batch_size -----------------------------------#
f1 = open("learning_rate.txt", "r")
CELSGD1 = [[], []]
CELAdam1 = [[], []]
CELAG1 = [[], []]


for line in f1:
    splitStr = line.split(' ')
    xval = float(splitStr[4])
    yval = float(splitStr[5])

    if splitStr[1] == 'SGD':
        CELSGD1[0].append(xval)
        CELSGD1[1].append(yval)
    elif splitStr[1] == 'Adam':
        CELAdam1[0].append(xval)
        CELAdam1[1].append(yval)
    else:
        CELAG1[0].append(xval)
        CELAG1[1].append(yval)

f1.close()

plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.scatter(CELSGD1[0], CELSGD1[1])
plt.scatter(CELAdam1[0], CELAdam1[1])
plt.scatter(CELAG1[0], CELAG1[1])
plt.legend(['CEL, SGD', 'NLLLoss, SGD', 'CEL, Adam', 'NLLLoss, Adam', 'CEL, AdaGrad', 'NLLLoss, AdaGrad'],
           loc='best')
plt.show()

