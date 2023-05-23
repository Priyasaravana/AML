import os
import os.path as osp
import sys
import torch
import pickle
import numpy as np
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from torch import optim
from torch.utils.data import DataLoader
from src import models
from args import argument_parser, dataset_kwargs, optimizer_kwargs
from src.utils.avgmeter import AverageMeter
from src.utils.loggers import Logger
from src.utils.generaltools import set_random_seed
from src.utils.iotools import check_isfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import clsHMDBModel as hmdbModel
from src import models
from clsLoadDataset import getDataLoader
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
# global variables
parser = argument_parser()
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    #if not args.use_avai_gpus:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    #use_gpu = torch.cuda.is_available()
    #if args.use_cpu:
    #    use_gpu = False
    log_name = "log_test.txt" 
    save_dir = os.getcwd()
    sys.stdout = Logger(osp.join(save_dir, log_name))
    print(f"==========\nArgs:{args}\n==========")
    # get the dataset and data loader
    ds = getDataLoader()
    train_set, test_set = train_test_split(ds, test_size=0.2, random_state=42)

    model = models.init_model(name='vitb16')

    learning_rate = args.lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    train_loader = DataLoader(train_set, shuffle=True, batch_size=train_batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=test_batch_size)
    train = 'yes'
    num_epoch = args.max_epoch
    if train == 'no':
        for epoch in range(num_epoch):
            print(f"Epoch: {epoch+1}/{num_epoch}")
            avg_loss, label_s, pred_s, logits_s = hmdbModel.train(model, train_loader, optimizer, device)
            
            accuracy = accuracy_score(torch.cat(label_s).cpu().numpy(), torch.cat(pred_s).cpu().numpy())    
            top1_accuracy = accuracy_score(torch.cat(label_s).cpu().numpy(), torch.cat(pred_s).cpu().numpy())
            logits_s = np.concatenate([logits.detach().cpu().numpy() for logits in logits_s])  # Convert list of tensors to numpy array
            top5_accuracy = hmdbModel.topk_accuracy(np.concatenate(label_s), logits_s, k=5)
            #writer.add_scalar('training_loss', avg_loss, global_step = epoch)
            print("Accuracy:", accuracy)
            print("Top-1 Accuracy:", top1_accuracy)
            print("Top-5 Accuracy:", top5_accuracy)
    else:
        model = torch.load('2E_8B_trained_model.pt')
        label_test, pred_test, logits_test = hmdbModel.test(model, test_loader, device)

    # Calculate and plot confusion matrix
    if train == 'yes':
        cf_max = confusion_matrix(torch.cat(pred_s).cpu().numpy(), torch.cat(label_s).cpu().numpy())
        sns.heatmap(cf_max, annot=True, fmt='g')

        # Save the model trained on the first dataset
        torch.save(model.state_dict(), "trained_model.pt")    
        cr = classification_report(torch.cat(pred_s).cpu().numpy(), torch.cat(label_s).cpu().numpy())
        print(cr)
    else:
        # Save the model trained on the first dataset          
        accuracy = accuracy_score(torch.cat(label_test).cpu().numpy(), torch.cat(pred_test).cpu().numpy())    
        top1_accuracy = accuracy_score(torch.cat(label_test).cpu().numpy(), torch.cat(pred_test).cpu().numpy())
        logits_s = np.concatenate([logits.detach().cpu().numpy() for logits in logits_s])  # Convert list of tensors to numpy array
        top5_accuracy = hmdbModel.topk_accuracy(np.concatenate(label_test), logits_s, k=5)
        cf_max = confusion_matrix(torch.cat(pred_test).cpu().numpy(), torch.cat(label_test).cpu().numpy())
        sns.heatmap(cf_max, annot=True, fmt='g')
        cr = classification_report(torch.cat(pred_test).cpu().numpy(), torch.cat(label_test).cpu().numpy())
        print(cr)

