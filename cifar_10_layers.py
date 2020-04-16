import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters
input_size = 3 * 32 * 32
num_classes = 10

num_epochs = 10
batch_size = 32
values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]

for j in range(4):
    for num in values:
        learning_rate = num * (10**j)


        # CIFAR_10 dataset (images and labels)
        train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        test_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                                  train=False,
                                                  transform=transforms.ToTensor())

        # Data loader (input pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        # Logistic regression model

        class Model(torch.nn.Module):
            def __init__(self, input_size, no_of_classes):
                super(Model, self).__init__()
                self.ip = input_size
                self.nb_class = no_of_classes

                self.h1 = nn.Linear(self.ip, 512)
                self.h2 = nn.Linear(512, self.nb_class)

                self.relu = nn.ReLU()

            def forward(self, X):
                h1 = self.relu(self.h1(X))
                op = self.h2(h1)
                return op

        # model = nn.Linear(input_size, num_classes)
        model = Model(input_size, num_classes)
        # Loss annn.Lid optimizer
        # nn.CrossEntropyLoss() computes softmax internally
        criterion = nn.CrossEntropyLoss()  # Read About this Loss + Try Other Losses
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate) # Read About This Try different opt

        strLoss = "CrossEntropyLoss"
        strOptimizer = "Ada"

        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Reshape images to (batch_size, input_size)
                images = images.reshape(-1, input_size)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, input_size)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total
            print('Accuracy of the model on the 10000 test images: {} %'.format(accuracy))

        # Plotting:
        # 1. Append data to text file:
        #    loss, optimizer, batch_size, num_epochs, learning_rate, accuracy
        # 2. Plot using data in the file

        f = open("learning_rate.txt", "a")
        f.write("{} {} {} {} {} {}%\n".format(strLoss, strOptimizer, batch_size, num_epochs, learning_rate, accuracy))
        f.close()

        # Save the model checkpoint
        torch.save(model.state_dict(), 'model.ckpt')