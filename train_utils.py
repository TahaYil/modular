
import os
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def train_multi_class_model(model : nn.Module,
                            train_loader: DataLoader,
                            test_loader: DataLoader,
                            loss : Callable,
                            optimizer : torch.optim.Optimizer,
                            scheduler : torch.optim.lr_scheduler.StepLR ,
                            epochs : int,
                            patience : int = 2,
                            device : torch.device = "cpu") :
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_model_state_dict = None
    best_val_loss = float("inf")
    
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        run_loss , run_acc= 0.0 , 0.0
        total = 0

        for X , y in train_loader:
            total += y.size(0)
            X , y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            
            loss_value = loss(y_pred, y)
            loss_value.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)#Gradiant clipping
            #büyük gradiantları küçültür
            optimizer.step()
            run_loss += loss_value.item() * X.size(0) # ağırlıklara göre hesap
            
            y_pred_prob=F.softmax(y_pred,dim=1)  # extra probability 
            
            run_acc += multi_class_acc(y,y_pred) * y.size(0) # çarpma nedeni batch lerin 
                                                            # ağırlıklarına göre hareket etmek
            
        train_losses.append(run_loss/total)
        train_accuracies.append(run_acc/total)
        
        model.eval()
        #Validation
        
        val_loss , val_acc= 0.0 , 0.0
        total = 0
        with torch.no_grad():
            for X , y in test_loader:
                X , y = X.to(device), y.to(device)
                total += y.size(0)
                y_pred = model(X)
                val_loss += loss(y_pred, y).item() * X.size(0)
                val_acc += multi_class_acc(y,y_pred) * y.size(0)
        
        test_losses.append(val_loss/total)
        test_accuracies.append(val_acc/total)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]*100:.2f}%')
        print(f'Val Loss: {test_losses[-1]:.4f}, Val Acc: {test_accuracies[-1]*100:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        
        if(best_val_loss > val_loss):
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            counter = 0
            torch.save(best_model_state_dict, os.path.join(save_dir, "best_model.pth"))
            
        else:
            counter += 1
            if counter >= patience:
                print ("Early stopping triggered")
                break
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        
    return train_losses, train_accuracies, test_losses, test_accuracies
            
def multi_class_acc(y_true, y_pred):
    y_pred_index = torch.argmax(y_pred, dim=1)
    acc = (y_pred_index == y_true).sum().float() / y_true.size(0)
    return acc         
            
            
            
            
        
        

