import matplotlib.pyplot as plt

# ----------------------- MNIST ADVERSARIAL TRAINING -------------------------#
f1 = open("mnist_adversarial.txt", "r")

numEpochs = []
lossArr = []

for line in f1:
    splitStr = line.split(" ")
    numEpochs.append(int(splitStr[1]))
    lossArr.append(float(splitStr[2]))

f1.close()

plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("MNIST Adversarial Training")
plt.plot(numEpochs, lossArr)
plt.savefig("MNIST_adversarial.png")
plt.show()

# ----------------------- CIFAR-10 ADVERSARIAL TRAINING -------------------------#
f = open("cifar_10_adversarial.txt", "r")

numEpochs.clear()
lossArr.clear()

for line in f:
    splitStr = line.split(" ")
    numEpochs.append(int(splitStr[1]))
    lossArr.append(float(splitStr[2]))

f.close()

plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("CIFAR-10 Adversarial Training")
plt.plot(numEpochs, lossArr)
plt.savefig("CIFAR-10_adversarial.png")
plt.show()

# ----------------------- MNIST ROBUST LEARNING -------------------------#
f2 = open("mnist_robust_learning.txt", "r")

numEpochs.clear()
lossArr.clear()

for line in f2:
    splitStr = line.split(" ")
    numEpochs.append(int(splitStr[1]))
    lossArr.append(float(splitStr[2]))

f2.close()

plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("MNIST Robust Learning")
plt.plot(numEpochs, lossArr)
plt.savefig("MNIST_robust_learning.png")
plt.show()

# ----------------------- CIFAR-10 ROBUST LEARNING -------------------------#
f2 = open("cifar_10_robust_learning.txt", "r")

numEpochs.clear()
lossArr.clear()

for line in f2:
    splitStr = line.split(" ")
    numEpochs.append(int(splitStr[1]))
    lossArr.append(float(splitStr[2]))

f2.close()

plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("CIFAR-10 Robust Learning")
plt.plot(numEpochs, lossArr)
plt.savefig("CIFAR-10_robust_learning.png")
plt.show()

