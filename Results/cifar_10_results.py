import matplotlib.pyplot as plt
import numpy as np
import array

# ----------------------- accuracy as a result of change in learning rate, CIFAR-10 NO LAYERS -------------------------#
f1 = open("cifar_10_no_layers.txt", "r")
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
plt.title('CIFAR-10 Linear Regression')
plt.scatter(CELSGD1[0], CELSGD1[1])
plt.scatter(CELAdam1[0], CELAdam1[1])
plt.scatter(CELAG1[0], CELAG1[1])
plt.legend(['CEL, SGD', 'CEL, Adam', 'CEL, AdaGrad'],
           loc='best')
plt.savefig("CIFAR_10_log_regression.png")
plt.show()


# ----------------------- accuracy as a result of change in learning rate, MNIST w/ LAYERS ------------------------#
f2 = open("cifar_10_layers.txt", "r")
CELSGD2 = [[], []]
CELAdam2 = [[], []]
CELAG2 = [[], []]


for line in f2:
    splitStr = line.split(' ')
    xval = float(splitStr[4])
    yval = float(splitStr[5])

    if splitStr[1] == 'SGD':
        CELSGD2[0].append(xval)
        CELSGD2[1].append(yval)
    elif splitStr[1] == 'Adam':
        CELAdam2[0].append(xval)
        CELAdam2[1].append(yval)
    else:
        CELAG2[0].append(xval)
        CELAG2[1].append(yval)

f2.close()

plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('CIFAR-10 with Layers')
plt.scatter(CELSGD2[0], CELSGD2[1])
plt.scatter(CELAdam2[0], CELAdam2[1])
plt.scatter(CELAG2[0], CELAG2[1])
plt.legend(['CEL, SGD', 'CEL, Adam', 'CEL, AdaGrad'],
           loc='best')
plt.savefig("CIFAR_10_layers.png")
plt.show()

# ----------------------- accuracy as a result of change in learning rate, MNIST CNN ------------------------#
f3 = open("cifar_10_cnn.txt", "r")

numEpochs = []
lossArr = []
accuracyArr = []

for line in f3:
    splitStr = line.split(" ")
    numEpochs.append(int(splitStr[0]))
    lossArr.append(float(splitStr[1]))
    accuracyArr.append(float(splitStr[2]))

f3.close()

plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("CIFAR-10 Convolution Neural Network")
plt.scatter(numEpochs, accuracyArr)
plt.savefig("CIFAR_10_CNN_accuracy")
plt.show()

for num in lossArr:
    num = "{:.5f}".format(num)

plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title('CIFAR-10 Convolution Neural Network')
plt.scatter(numEpochs, lossArr)
plt.savefig("CIFAR_10_CNN_loss")
plt.show()

