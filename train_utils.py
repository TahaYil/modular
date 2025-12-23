
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
                            device : torch.device = "cpu",
                            model_name : str="multi_class_model"):
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
            print("Best model updated")
            save_path_zibab = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(best_model_state_dict, save_path_zibab)


        else:
            counter += 1
            if counter >= patience:
                print ("Early stopping triggered")
                break
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)

    torch.save(best_model_state_dict, os.path.join(save_dir, "best_model.pth"))
    print("Training complete. Best model saved.")
    return train_losses, train_accuracies, test_losses, test_accuracies




def train_multi_class_one_hot(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss: Callable,                     # BCEWithLogitsLoss
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epochs: int,
    patience: int = 5,
    device: str = "cpu",
        model_name : str="multi_class_one_hot_model"
):
    model.to(device)

    num_classes = model.fc.out_features  # 🔥 otomatik sınıf sayısı

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # ==========================
        # TRAIN
        # ==========================
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y_int = y.long().to(device)  # (B,)

            batch_size = y_int.size(0)
            total_samples += batch_size

            # ---- ONE HOT ----
            y_onehot = torch.zeros(
                batch_size, num_classes, device=device
            )
            y_onehot.scatter_(1, y_int.unsqueeze(1), 1)

            optimizer.zero_grad()

            logits = model(X)  # (B, C)

            loss_value = loss(logits, y_onehot)
            loss_value.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss_value.item() * batch_size

            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y_int).sum().item()

        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ==========================
        # VALIDATION
        # ==========================
        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                y_int = y.long().to(device)

                batch_size = y_int.size(0)
                val_total += batch_size

                y_onehot = torch.zeros(
                    batch_size, num_classes, device=device
                )
                y_onehot.scatter_(1, y_int.unsqueeze(1), 1)

                logits = model(X)

                loss_value = loss(logits, y_onehot)
                val_running_loss += loss_value.item() * batch_size

                preds = torch.argmax(logits, dim=1)
                val_running_correct += (preds == y_int).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_running_correct / val_total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        # ==========================
        # LOG
        # ==========================
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

        # ==========================
        # EARLY STOPPING
        # ==========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            save_path_zibab = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(best_model_state, save_path_zibab)
            early_stop_counter = 0
            print("✅ Best model updated")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("⏹ Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("🎯 Training complete. Best model loaded.")
    return train_losses, train_accs, val_losses, val_accs




def multi_class_acc(y_true, y_pred):
    y_pred_index = torch.argmax(y_pred, dim=1)
    acc = (y_pred_index == y_true).sum().float() / y_true.size(0)
    return acc         
            
            
            
            
        
        

