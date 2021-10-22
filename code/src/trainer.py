import torch
import os
import json
from tqdm import tqdm
import subprocess

from src.utils.utils import pyt2np, AverageMeter
import copy


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        data_loader_train,
        data_loader_val,
        # data_loader_test,
        exp_dir,
        dev,
        logger,
        n_epochs,
    ):

        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        # self.data_loader_test = data_loader_test
        self.model = model
        self.opt = optimizer
        self.scheduler = scheduler
        self.dev = dev
        self.print_freq = 100  # Update print frequency
        self.save_freq = 10
        self.exp_dir = exp_dir
        self.num_train_epochs = n_epochs
        self.logger = logger
        self.criterion_vertices = torch.nn.L1Loss().cuda(self.dev)
        self.log_losses = AverageMeter()
        self.log_loss_vertices = AverageMeter()


    def vertices_loss(self, pred_vertices, gt_vertices):
        return self.criterion_vertices(pred_vertices, gt_vertices)


    def one_pass(self, data_loader, phase, preds_storage=None, epoch=None):
        """
        Performs one pass on the entire training or validation dataset
        """
        model = self.model
        opt = self.opt
        scheduler = self.scheduler

        self.log_losses.reset()
        self.log_loss_vertices.reset()
        for it, batch in enumerate(data_loader):
            # Zero the gradients of any prior passes
            opt.zero_grad()
            # Send batch to device
            self.send_to_device(batch)

            gt_vertices = batch['verts']

            # Forward pass
            pred_vertices = model(batch['image'])

            if phase == 'train':
                # compute 3d vertex loss
                loss_vertices = self.vertices_loss(pred_vertices, gt_vertices)
                loss = loss_vertices
                batch_size = batch['image'].shape[0]
                self.log_loss_vertices.update(loss_vertices.item(), batch_size)
                self.log_losses.update(loss.item(), batch_size)

                # Backward pass, compute the gradients
                loss.backward()
                opt.step()
                self.adjust_learning_rate(opt, epoch)

                if (it % self.print_freq) == 0 or it == len(data_loader)-1:
                    str_print = ""
                    if epoch is not None:
                        str_print += f"Epoch: {epoch:03d}\t"
                    str_print += f"Iter: {it:04d}/{len(data_loader):04d}  "
                    str_print += f"loss: {self.log_losses.avg:.10f}  "
                    str_print += f" V loss: {self.log_loss_vertices.avg:.10f}  "
                    str_print += f"lr: {opt.param_groups[0]['lr']:.6f}  "
                    self.logger.info(str_print)


        if phase == 'train' and it == len(data_loader)-1:
            self.logger.info(f"***** Epoch {epoch:03d} mean loss: {self.log_losses.avg:.10f} *****\t")

        return self.log_losses.avg # , preds_storage, preds_storage_reg


    def adjust_learning_rate(self, opt, epoch):
        lr = 0.0001 * (0.1 ** (epoch // 100))
        for param_group in opt.param_groups:
            param_group['lr'] = lr


    def send_to_device(self, batch):
        for k, v in batch.items():
            batch[k] = v.to(self.dev)


    def train_model(self):

        loss = float('inf')
        for e in range(self.num_train_epochs):
            self.logger.info(f"\nEpoch: {e+1:04d}/{self.num_train_epochs:04d}")
            # Train one epoch
            self.model.train()
            self.logger.info("##### TRAINING #####")
            self.one_pass(self.data_loader_train, phase="train", epoch=e)
            # Evaluate on validation set
            with torch.no_grad():
                self.model.eval()
                self.logger.info("##### EVALUATION #####")
                loss_tot = self.one_pass(self.data_loader_val, phase="eval", epoch=e)
                if loss_tot < loss:
                    loss = loss_tot
                    best_loss = loss
                    best_model_state = copy.deepcopy(self.model.state_dict())

            if (e % self.save_freq) == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.exp_dir, f"model_{e:04d}_{loss:.10f}.pt"),
                )
            if not os.path.exists(os.path.join(self.exp_dir, f"bestmodel_{e:04d}_{best_loss:.10f}.pt")):
                torch.save(best_model_state, os.path.join(self.exp_dir, f"bestmodel_{e:04d}_{best_loss:.10f}.pt"))

        torch.save(
            self.model.state_dict(), os.path.join(self.exp_dir, f"model_last.pt")
        )


    '''def test_model(self):
        """
        Runs model on testing data
        """
        print("##### TESTING #####")
        # NOTE If you are saving the best performing model, you may want to first load
        # the its weights before running test
        with torch.no_grad():
            _, preds_storage = self.one_pass(
                self.data_loader_test, phase="test", preds_storage=[]
            )

        preds = pyt2np(torch.cat(preds_storage, dim=0)).tolist()

        test_path = os.path.join(self.exp_dir, "test_preds.json")
        print(f"Dumping test predictions in {test_path}")
        with open(test_path, "w") as f:
            json.dump(preds, f)
        subprocess.call(['gzip', test_path])'''


