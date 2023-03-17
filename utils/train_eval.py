from tqdm import tqdm 
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import confusion_matrix
import torch.utils.tensorboard
import utils


class Trainer(nn.Module):

    def __init__(self,  model, optimizer, criterion, train_loader, 
        valid_loader, train_set, epochs, scheduler=None, sequence=True, tboard_name=None, start_epoch=0, 
        all_labels=None, print_intermediate_vals=False, model_name="") -> None:
        super().__init__()
        
        # needed for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.sequence = sequence

        if train_set == "coco":
            self.train_fn = self.coco_train
        elif train_set == "cityscapes":
            self.train_fn = self.cityscapes_train


        if scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        else:
            self.scheduler = scheduler

        # loading/saving model
        self.model_name = model_name

        # needed for plotting the losses and other metrics
        if tboard_name == None:
            self.tboard = utils.make_tboard_logs(f"{self.model.__class__.__name__}")
        else:
            self.tboard = utils.make_tboard_logs(tboard_name)

        self.all_labels = all_labels
        self.print_intermediate_vals = print_intermediate_vals
        self.start_epoch = start_epoch

        # losses
        self.train_loss = []
        self.val_loss =  []
        self.loss_iters = []
        self.valid_mIoU = []
        self.conf_mat = None

    def train_epoch(self, current_epoch):
        """ Training a model for one epoch """
        
        loss_list = []
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, (images, labels) in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device).to(torch.long)

            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()
            
            preds, loss = self.train_fn(images, labels)
            
            # Calculate Loss: softmax --> cross entropy loss
            loss_list.append(loss.item())
            
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            # Updating parameters
            self.optimizer.step()
            
            progress_bar.set_description(f"Epoch {current_epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")

            if i == len(self.train_loader)-1:
                mean_loss = np.mean(loss_list)
                progress_bar.set_description(f"Epoch {current_epoch+1} Iter {i+1}: mean loss {mean_loss.item():.5f}. ")

        return mean_loss, loss_list


    @torch.no_grad()
    def eval_model(self):
        """ Evaluating the model for either validation or test """
        correct = 0
        total = 0
        loss_list = []
        epsilon = 1e-6

        if self.all_labels != None:
            self.conf_mat = torch.zeros(self.all_labels, self.all_labels)
        else:
            self.conf_mat == None
        
        for images, labels in self.valid_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
        

            outputs, loss = self.train_fn(images, labels)
        
            loss_list.append(loss.item())

            preds = torch.argmax(outputs, dim=2)
            # mIoU
            labels = labels.squeeze().view(-1)
            preds = preds.squeeze().view(-1)

            if self.all_labels!= None:
                self.conf_mat += confusion_matrix(
                    y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), 
                    labels=np.arange(0, self.all_labels, 1)
                )

        iou = self.conf_mat.diag() / (self.conf_mat.sum(axis=1) + self.conf_mat.sum(axis=0) - self.conf_mat.diag() + epsilon)
        mIoU = iou.mean()
        loss = np.mean(loss_list)
        return mIoU, loss


    def train_model(self):
        """ Training a model for a given number of epochs"""
        
        start = time.time()
        self.model = self.model.to(self.device)
        
        for epoch in range(self.epochs):
            
            # validation epoch
            self.model.eval()  # important for dropout and batch norms
            mIoU, loss = self.eval_model()
            self.valid_mIoU.append(mIoU)
            self.val_loss.append(loss)

            # if we want to use tensorboard
            if self.tboard !=None:
                self.tboard.add_scalar(f'mIoU/Valid', mIoU, global_step=epoch+self.start_epoch)
                self.tboard.add_scalar(f'Loss/Valid', loss, global_step=epoch+self.start_epoch)
            
            # training epoch
            self.model.train()  # important for dropout and batch norms
            mean_loss, cur_loss_iters = self.train_epoch(epoch)
            self.scheduler.step(self.val_loss[-1])
            self.train_loss.append(mean_loss)

            # if we want to use tensroboard
            if self.tboard != None:
                self.tboard.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+self.start_epoch)

            self.loss_iters = self.loss_iters + cur_loss_iters
            
            if self.print_intermediate_vals: # and epoch % 5 == 0 or epoch==self.epochs-1):
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"    Train loss: {round(mean_loss, 5)}")
                print(f"    Valid loss: {round(loss, 5)}")
                print(f"    mIoU: {mIoU}%")
                print("\n")

            self.save_model(self.start_epoch + epoch)
        
        end = time.time()
        print(f"Training completed after {(end-start)/60:.2f}min")

    def save_model(self, current_epoch):
        utils.save_model(
            self.model, 
            self.optimizer, 
            current_epoch,
            [self.train_loss, self.val_loss, self.loss_iters, self.valid_mIoU, self.conf_mat],
            self.model_name
            )
            

    def load_model(self):
        self.model, self.optimizer, self.start_epoch, self.stats = utils.load_model(
            self.model, 
            self.optimizer, 
            f"models/checkpoint_{self.model.__class__.__name__}{self.model_name}_epoch_{self.start_epoch - 1}.pth",
            self.device
        )
        self.train_loss, self.val_loss, self.loss_iters, self.valid_mIoU, self.conf_mat = self.stats

    def count_model_params(self):
        """ Counting the number of learnable parameters in a nn.Module """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return num_params

    def coco_train(self, images, labels):
        # sequence if necessary for single images
        outputs = self.model(images.unsqueeze(1))

        print(outputs.shape, labels.shape)
        print(outputs.shape)
        # preds = torch.argmax(outputs, dim=2)

        print(outputs.squeeze().shape, labels.squeeze().shape)
        loss = self.criterion(outputs.squeeze(), labels.squeeze().long())

        return outputs, loss

    def cityscapes_train(self, images, labels):
        # sequence if necessary for single images
        if self.sequence == True:
            # Forward pass only to get logits/output
            outputs = self.model(images)
        else:
            outputs = self.model(images.unsqueeze(1))
        
        # dataset is given with 
        gt_idx = labels[0]
        assert gt_idx.size() == (1,), f"Expected to be the index of ground truth, got f{gt_idx} instead."
        labels = labels[1]

        loss = self.criterion(outputs[:, gt_idx].squeeze(), labels.squeeze())

        return outputs, loss
