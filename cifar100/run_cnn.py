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
    num_filters = 2*128*8
    epochs = 200
    bias = False
    loss_fn = 'crossentropy'
    opt = 'sgd'
    zca = True
    augmentation = None #'flips'
    parallel = True
    half = False
    batch = False
    best_acc = 0.0

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
        nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=bias),
        # nn.BatchNorm2d(num_filters),
        activation(),
        nn.AvgPool2d(kernel_size=8, stride=8),
        Normalize(),
        Flatten(),
        nn.Linear(num_filters, 100, bias=bias)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and parallel:
        print('Multiple GPUs')
        net = nn.DataParallel(net)
    net = net.to(device)
    dtype=torch.float32
    if half:
        net = net.half()
        dtype=torch.float16

    square_loss = False
    if loss_fn == "mse":
        square_loss = True
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if opt == 'sgd':
        lr_init = .01
        optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=0.0005 , nesterov=True)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 120], gamma=0.2, last_epoch=-1)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    batch_size = 128
    num_classes = 100
    print_every = 1

    trainset = torchvision.datasets.CIFAR100('./data', train=True, transform=None, target_transform=None, download=True)
    testset = torchvision.datasets.CIFAR100('./data', train=False, transform=None, target_transform=None, download=True)
    (X_train, X_test), global_ZCA = utils.preprocess(trainset.data, testset.data, min_divisor=1e-8, zca_bias=1e-4, return_weights=True)

    print('Got Data')
    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    class CustomTensorDataset(torch.utils.data.Dataset):
        """TensorDataset with support of transforms.
        """
        def __init__(self, tensors, targets, transform=None):
            assert tensors.size(0) == targets.size(0)
            self.tensors = tensors
            self.targets = targets
            self.transform = transform

        def __getitem__(self, idx):
            x = self.tensors[idx]

            if self.transform == 'flips':
                if np.random.binomial(1,0.5) == 1:
                    x = x.flip(2)
            y = self.targets[idx]

            return x, y

        def __len__(self):
            return self.tensors.size(0)

    X_train = np.transpose(X_train, (0,3,1,2))
    X_test = np.transpose(X_test, (0,3,1,2))
    if augmentation == 'flips':
        trainset = CustomTensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long), transform=augmentation)
        testset = CustomTensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    else:
        trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.tensor(y_train, dtype=torch.long))
        testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.tensor(y_test, dtype=torch.long))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=int(batch_size/2),
                                             shuffle=False, num_workers=0)

    best_model_wts = copy.deepcopy(net.state_dict())

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
    if batch:
        batch_str = '_batch'
    else:
        batch_str = ''

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        train_correct = 0
        train_total = 0
        train_bar = tqdm.tqdm(total=int(50000/batch_size), desc='Epoch ' + str(epoch), position=0)
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
            if testacc >= best_acc:
                best_acc = testacc
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save(best_model_wts, 'weights-100/myrtle10' + str(opt) + '_' + str(loss_fn) + '_' + str(num_filters) + zca_str + aug_str + half_str + batch_str + '_best.pt')
            print("Best so far: " + str(best_acc))

    torch.save(copy.deepcopy(net.state_dict()), 'weights-100/myrtle10' + str(opt) + '_' + str(loss_fn) + '_' + str(num_filters) + '_' + str(epochs) + 'ep' + zca_str + aug_str + half_str + batch_str + '.pt')
    net.load_state_dict(best_model_wts)

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
