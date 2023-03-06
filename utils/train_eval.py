from tqdm import tqdm 
import numpy as np
import torch
import time
from sklearn.metrics import confusion_matrix
import torch.utils.tensorboard


def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """
    
    loss_list = []
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(images)
         
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
        progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")

        if i == len(train_loader)-1:
            mean_loss = np.mean(loss_list)
            progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: mean loss {mean_loss.item():.5f}. ")

    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, all_labels=None):
    """ Evaluating the model for either validation or test """
    correct = 0
    total = 0
    loss_list = []

    if all_labels != None:
        conf_mat = torch.zeros((len(all_labels), len(all_labels)))
    else:
        conf_mat == None
    
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass only to get logits/output
        outputs = model(images)
                 
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
            
        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len( torch.where(preds==labels)[0] )
        total += len(labels)

        #confusion matrix
        if all_labels!= None:
            conf_mat += confusion_matrix(
                y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), 
                labels=np.arange(0, len(all_labels), 1)
                )

    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    return accuracy, loss, conf_mat


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def train_model(
    model, optimizer, scheduler, criterion, train_loader, 
    valid_loader, num_epochs, device, tboard=None, start_epoch=0, 
    all_labels=None, print_intermediate_vals=False
    ):
    """ Training a model for a given number of epochs"""
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []
    start = time.time()
    
    for epoch in range(num_epochs):
           
        # validation epoch
        model.eval()  # important for dropout and batch norms
        accuracy, loss, conf_mat = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device, all_labels=all_labels
            )
        valid_acc.append(accuracy)
        val_loss.append(loss)

        # if we want to use tensorboard
        if tboard !=None:
            tboard.add_scalar(f'Accuracy/Valid', accuracy, global_step=epoch+start_epoch)
            tboard.add_scalar(f'Loss/Valid', loss, global_step=epoch+start_epoch)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device,
            )
        scheduler.step()
        train_loss.append(mean_loss)

        # if we want to use tensroboard
        if tboard != None:
            tboard.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)

        loss_iters = loss_iters + cur_loss_iters
        
        if(epoch % 5 == 0 or epoch==num_epochs-1) and print_intermediate_vals:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")
            print(f"    Accuracy: {accuracy}%")
            print("\n")
    
    end = time.time()
    print(f"Training completed after {(end-start)/60:.2f}min")
    return train_loss, val_loss, loss_iters, valid_acc, conf_mat


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params