import torch
import numpy as np
import torch.nn.functional as F
import time
import csv
from sklearn.metrics import f1_score


def output_to_class(output):
    """
    takes the output from a nn feeded with a batch and returns the predicted classes
    """
    
    classes = []
    for sample in output:
        classes.append(list(sample).index(max(sample)))
    return classes


# This function trains the model for one epoch
def train(args, model, device, train_loader, optimizer, epoch, start_time):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1: {:.4f}\tRuntime: {:.1f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), f1_score(target.detach().cpu().numpy(), output_to_class(output), average='micro'), time.time() - start_time))
                
                
# This function evaluates the model on the test data
def test(args, model, device, test_loader, epoch, trainDataset, testDataset, path_save):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        familyPredictions = np.zeros(len(test_loader.dataset), dtype=np.int)
        for index, samples in enumerate(test_loader):
            samples = samples.to(device)
            familyPredictions[index*len(samples):(index+1)*len(samples)] = model(samples).max(1)[1].cpu() # get the index of the max log-probability
    
    familyPredictionStrings = trainDataset.transformInstrumentsFamilyToString(familyPredictions.astype(int))

    with open(path_save + 'NN-submission-' +str(epoch)+'.csv', 'w', newline='') as writeFile:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=', ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for index in range(len(testDataset)):
            writer.writerow({'Id': index, 'Predicted': familyPredictionStrings[index]})
    print('saved predictions')