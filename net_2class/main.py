import torch
import torch.nn as nn
import torch.optim as optim
from model import CNN
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data_input=pd.read_csv('../data/preprocess_dataset/dataset_2class_3descriptor.csv')


x = data_input.iloc[:, 1:-1]

y = data_input.iloc[:, -1]


scaler = MinMaxScaler()


X = scaler.fit_transform(x)
inputs = torch.tensor(X, dtype=torch.float)
labels = torch.tensor(y, dtype=torch.long)

train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2,shuffle=True, random_state=1)

train_dataset = TensorDataset(train_inputs, train_labels)
test_dataset = TensorDataset(test_inputs, test_labels)





def train(net, criterion, optimizer, trainloader, validloader, epochs,fold):
    x = []
    t_loss = []  # train_loss
    v_loss = []  # val_loss
    t_acc = []  # train_acc
    v_acc = []  # val_acc
    for epoch in range(epochs):
        total=0
        correct=0
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy =correct / total
        print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

        x.append(epoch+1)
        t_loss.append(running_loss / len(trainloader))
        t_acc.append(accuracy)


        total = 0
        correct = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy =correct / total
            print('Validation loss on epoch %d: %.3f' % (epoch + 1, running_loss / len(validloader)))
            v_loss.append(running_loss/ len(validloader))
            v_acc.append(accuracy)


    np.savetxt('%d_t_loss.txt' %(fold+1), t_loss, fmt='%f', delimiter=',')
    np.savetxt('%d_v_acc.txt' %(fold+1), v_acc, fmt='%f', delimiter=',')
    np.savetxt('%d_t_acc.txt' %(fold+1), t_acc, fmt='%f', delimiter=',')
    np.savetxt('%d_v_loss.txt' %(fold+1), v_loss, fmt='%f', delimiter=',')
    #plt.figure()
    plt.figure(figsize=(9, 7))
    plt.style.use('default')
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.xticks(size=24)
    plt.yticks(size=24)
    font2 = {'size': 28}
    font3 = {'size': 13}

    plt.plot(x, t_loss, 'r', lw=3)
    plt.plot(x, v_loss, 'b', lw=3)
    plt.plot(x, t_acc, 'y', lw=3)
    plt.plot(x, v_acc, 'g', lw=3)

    plt.xlabel("Epochs", font2)

    plt.legend(["Train loss",
                "Validate loss",
                "Train accuracy",
                "Validate accuracy"], loc='lower left', prop=font3)
    plt.savefig('%d _fold_cross_validation_ACC.tif' % (fold+1), dpi=300)

def test(net, testloader):
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())



#    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')


#    print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
    #print('Precision: {:.2f}'.format(precision))
    #print('Recall: {:.2f}'.format(recall))
    #print('F1 Score: {:.2f}'.format(f1))
    #print('Accuracy on test set: %d' % (accuracy))
    return accuracy, precision, recall, f1




def k_fold_cross_validation(net, criterion, optimizer, dataset, testset,k, epochs, batch_size):
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    accuracies=[]
    precisions = []
    recalls = []
    f1s = []
    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        # dataset set
        net = net.to(device)
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        # train
        #print('Training on fold %d...' % (fold + 1))
        train(net, criterion, optimizer, trainloader, validloader, epochs,fold)

        # test
        #print('Testing on fold %d...' % (fold + 1))
        accuracy, precision, recall, f1 = test(net, testloader)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    # average acc
    avg_accuracy = sum(accuracies) / k
    avg_precision = sum(precisions) / k
    avg_recall = sum(recalls) / k
    avg_f1 = sum(f1s) / k
    return avg_accuracy, avg_precision, avg_recall, avg_f1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0001
epochs = 100
batch_size = 64
k = 5
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
avg_accuracies = []
avg_precisions = []
avg_recalls = []
avg_f1s = []
for i in range(1):#multiple test
    avg_accuracy, avg_precision, avg_recall, avg_f1 = k_fold_cross_validation(net, criterion, optimizer, train_dataset,test_dataset, k, epochs, batch_size)
    avg_accuracies.append(avg_accuracy)
    avg_precisions.append(avg_precision)
    avg_recalls.append(avg_recall)
    avg_f1s.append(avg_f1)
'''
    with open("./results.txt", 'a') as f:
        f.write('Accuracy(std):{:.4f}'.format(avg_accuracy) + '\n')
        f.write('Precision(std):{:.4f}'.format(avg_precision) + '\n')
        f.write('Recall(std):{:.4f}'.format(avg_recall) + '\n')
        f.write('F1(std):{:.4f}'.format(avg_f1) + '\n')

    print('Accuracy(std):{:.4f}'.format(avg_accuracy))
    print('Precision(std):{:.4f}'.format(avg_precision))
    print('Recall(std):{:.4f}'.format(avg_recall))
    print('F1(std):{:.4f}'.format(avg_f1))
'''