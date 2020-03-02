import numpy as np
import tqdm
import copy
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import sys
sys.path.insert(0, "..")
import utils

def run_exp():
    num_filters = 128*8
    epochs = 20
    bias = False
    loss_fn = "mse"
    opt = 'adam'
    zca = False
    augmentation = None
    parallel = True
    half = False

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    class Normalize(nn.Module):
        def forward(self, input):
            eps=1e-8
            if half:
                eps = 1e-3
            return nn.functional.normalize(input, p=2, dim=1, eps=eps)

    activation = nn.ReLU

    net = nn.Sequential(
        nn.Conv2d(1, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        activation(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        activation(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        activation(),
        nn.AvgPool2d(kernel_size=7, stride=7),
        Normalize(),
        Flatten(),
        nn.Linear(num_filters, 10, bias=bias)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and parallel:
        print('Multiple GPUs')
        net = nn.DataParallel(net)
    net = net.to(device)
    dtype = torch.float32
    if half:
        net = net.half()
        dtype = torch.float16

    square_loss = False
    if loss_fn == "mse":
        square_loss = True
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if opt == 'sgd':
        lr_init = 0.1
        optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=0.0005 , nesterov=True)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120], gamma=0.1, last_epoch=-1)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    batch_size = 128
    num_classes = 10
    print_every = 1

    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size/2),
                                             shuffle=False, num_workers=0)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(net.state_dict())

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        train_correct = 0
        train_total = 0
        train_bar = tqdm.tqdm(total=int(60000/batch_size), desc='Epoch ' + str(epoch), position=0)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if half:
                inputs, labels = inputs.to(device).half(), labels.to(device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs).view(-1, num_classes)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            if square_loss:
                labels = torch.FloatTensor(outputs.size()).zero_().scatter_(1, labels.detach().cpu().reshape(outputs.size()[0], 1), 1).to(dtype=dtype, device=device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.update(1)
        if opt == 'sgd':
            lr_scheduler.step()

        if epoch % print_every == 0:
            print('Epoch: ' + str(epoch))
            net.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    if half:
                        inputs, labels = inputs.to(device).half(), labels.to(device)
                    else:
                        inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs).view(-1, num_classes)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    if square_loss:
                        labels = torch.FloatTensor(outputs.size()).zero_().scatter_(1, labels.detach().cpu().reshape(outputs.size()[0], 1), 1).to(dtype=dtype, device=device)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
            trainacc = train_correct/train_total*100
            testacc = test_correct/test_total*100
            print('Train Loss: ' + str(running_loss))
            print("Train Acc: " + str(trainacc))
            print('Test Loss: ' + str(test_loss))
            print("Test Acc: " + str(testacc))
            if testacc > best_acc:
                best_acc = testacc
                best_model_wts = copy.deepcopy(net.state_dict())
            print("Best so far: " + str(best_acc))

    if augmentation:
        aug_str = '_' + str(augmentation)
    else:
        aug_str = ''

    if zca:
        zca_str = '_zca'
    else:
        zca_str = ''
    if half:
        half_str = '_half'
    else:
        half_str = ''

    torch.save(copy.deepcopy(net.state_dict()), 'weights-mnist/myrtle5' + str(opt) + '_' + str(loss_fn) + '_' + str(num_filters) + '_' + str(epochs) + 'ep' + zca_str + aug_str + half_str + '.pt')
    net.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'weights-mnist/myrtle5' + str(opt) + '_' + str(loss_fn) + '_' + str(num_filters) + zca_str + aug_str + half_str + '_best.pt')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if half:
                images, labels = images.to(device).half(), labels.to(device)
            else:
                images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100*correct/total
    print('Test Acc:' + str(acc))

if __name__ == "__main__":
    run_exp()
