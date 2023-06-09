from tqdm.notebook import tqdm
import torch.nn.functional as F
import torch
from src.utils.avgmeter import AverageMeter
from torch import nn
import numpy as np

def train(model, data_loader, optimizer, device):
    # meter
    loss_meter = AverageMeter()
    # switch to train mode
    model.train()
    label_ls = []
    pred_ls = []
    logits_ls = []
    loss_ls = []
    tk = tqdm(data_loader, total=int(len(data_loader)), desc='Test', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
        # fetch the data
        frame, label = data[0], data[1]
        # transfer the data to the device (GPU)
        frame, label = frame.to(device), label.to(device)

        # compute the forward pass
        output = model(frame)        
        logits = output
        pred = logits.argmax(axis=1)
        # compute the loss function
        loss_this = nn.functional.cross_entropy(logits, label.long())
        # zero the gradients
        optimizer.zero_grad()
        # compute the backward pass
        loss_this.backward()
        # update the parameters
        optimizer.step()
        label_ls.append(label)
        pred_ls.append(pred)
        logits_ls.append(logits)
        loss_ls.append(loss_this)
        # update the loss meter
        loss_meter.update(loss_this.item(), label.shape[0])

    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))
    return loss_meter.avg, label_ls, pred_ls, logits_ls, loss_ls


##define test function
def test(model, data_loader, device):
    # meters
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    # switch to test mode
    label_test = []
    pred_test = []
    logits_test = []
    loss_test = [] 
    model.eval()
    tk = tqdm(data_loader, total=int(len(data_loader)), desc='Test', unit='frames', leave=False)
    for batch_idx, data in enumerate(tk):
        # fetch the data
        frame, label = data[0], data[1]
        # after fetching the data transfer the model to the 
        # required device, in this example the device is gpu
        # transfer to gpu can also be done by 
        frame, label = frame.to(device), label.to(device)
        # since we dont need to backpropagate loss in testing,
        # we dont keep the gradient
        label_test = []
        pred_test = []
        logits_test = []
        with torch.no_grad():
            output = model(frame)
        logits = output
        pred = logits.argmax(axis=1)
        
        # compute the loss function just for checking
        loss_this = F.cross_entropy(logits, label.long())     
        # update the loss and accuracy meter 
        label_test.append(label)
        pred_test.append(pred)
        logits_test.append(logits)
        loss_test.append(loss_this)
        loss_meter.update(loss_this.item(), label.shape[0])
    return loss_meter.avg, label_test, pred_test, logits_test, loss_test
    

def topk_accuracy(y_true, y_pred, k=5):
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=1)
    sorted_indices = np.argsort(y_pred, axis=1)[:, ::-1]
    topk_pred = sorted_indices[:, :k]
    topk_true = np.expand_dims(y_true, axis=1)
    matches = np.any(topk_pred == topk_true, axis=1)
    topk_accuracy = np.mean(matches)
    return topk_accuracy