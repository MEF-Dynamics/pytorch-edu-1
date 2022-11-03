import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from model import Net

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    
    
    net = Net()
    criterion = nn.CrossEntropyLoss() # loss function, cross entropy, for classification, softmax is applied in the loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # momentum is a technique to accelerate SGD in the relevant direction and dampens oscillations
    
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        print(f'Epoch {epoch + 1}')
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad() # zero the parameter gradients 

            outputs = net(inputs)
            loss = criterion(outputs, labels) # calculate loss
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print('Finished Training')