import torch
import shutil
import os
import json
from tqdm import tqdm
# import subprocess
from src.utils.utils import pyt2np, AverageMeter
import copy
import numpy as np
from multiprocessing import Pool
# from functools import partial
import cv2
# import trimesh
# from mesh_intersection.bvh_search_tree import BVH
import pdb

device = torch.device('cuda')

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        data_loader_train,
        data_loader_val,
        data_loader_test,
        # data_loader_image_loss,
        exp_dir,
        dev,
        logger,
        n_epochs,
        scheduler_step,
        read_intermediate,
        attention,
    ):

        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        # self.data_loader_image_loss = data_loader_image_loss
        self.model = model
        self.opt = optimizer
        self.dev = dev
        self.print_freq = 100  # Update print frequency
        self.save_freq = 50
        self.exp_dir = exp_dir
        self.num_train_epochs = n_epochs
        self.scheduler_step = scheduler_step
        self.logger = logger
        self.criterion_vertices = torch.nn.L1Loss().cuda(self.dev)
        self.reg_loss_cri = torch.nn.MSELoss().cuda(self.dev)
        # self.criterion_images = torch.nn.L1Loss().cuda(self.dev)
        self.log_losses = AverageMeter()
        self.log_loss_vertices = AverageMeter()
        self.log_loss_reg = AverageMeter()
        self.log_loss_intersec = AverageMeter()
        # self.log_loss_image = AverageMeter()
        self.blender_path = '/home/crl-5/Desktop/blender-2.93.5-linux-x64/blender'
        self.edge_idx = torch.from_numpy(np.loadtxt('/home/crl-5/Desktop/cloth_recon/mesh_edge_idx.txt').astype(int)[:5100]).to(torch.int64).cuda()
        self.read_intermediate = read_intermediate
        self.attention = attention
        # self.m = BVH(max_collisions=8)

    def vertices_loss(self, pred_vertices, gt_vertices):
        return self.criterion_vertices(pred_vertices, gt_vertices)

    # def image_loss(self, pred_images, gt_images):
    #     return self.criterion_images(pred_images, gt_images)
    
    def one_pass(self, data_loader, phase, preds_storage=None, epoch=None):
        """
        Performs one pass on the entire training or validation dataset
        """
        model = self.model
        opt = self.opt

        self.log_losses.reset()
        self.log_loss_vertices.reset()
        self.log_loss_reg.reset()
        self.log_loss_intersec.reset()
        # self.log_loss_image.reset()
        for it, batch in enumerate(data_loader):
            # Zero the gradients of any prior passes
            opt.zero_grad()
            # Send batch to device
            self.send_to_device(batch)

            gt_vertices = batch['verts']
            # TODO unset 1300th dim to zero!
            # gt_root = gt_vertices[:, 1300, :]
            # TODO do not make minus z value
            # gt_root[:, 2] = 0.0
            # gt_vertices = gt_vertices - gt_root[:, None, :]

            # Forward pass
            if phase == 'train':
                is_training = True
            else:
                is_training = False
            output = model(batch['image'], is_training, self.read_intermediate, self.attention)
            if not self.read_intermediate:
                if not self.attention:
                    pred_vertices = output
                else:
                    pred_vertices, attention_list = output
            else:
                pred_vertices = output[-1]
            # pred_vertices = output['verts']
            
            
            
            # image loss
            # TODO how to send nparr to another script? subprocess complains the arr string/bstr is embedded null type...
            '''
            basedir = '/home/crl-5/Desktop/cloth_recon/tmp/'
            if not os.path.exists(basedir):
                os.mkdir(basedir)

            # for idx, ele in enumerate(pyt2np(gt_vertices)):
            #     np.savetxt(basedir+'gt_%05d.txt'%idx, ele)
            for idx, ele in enumerate(pyt2np(pred_vertices)):
                np.savetxt(basedir+'pred_%05d.txt'%idx, ele)
            # with open(basedir+'gt.npy', 'wb') as f:
            #     np.save(f, pyt2np(gt_vertices))
            # f.close()

            # with open(basedir+'pred.npy', 'wb') as f:
            #     np.save(f, pyt2np(pred_vertices))
            # f.close()

            commands = []
            for i in range(gt_vertices.shape[0]):
                # command = self.blender_path + ' --background --python /home/crl-5/Desktop/cloth_recon/code/src/utils/renderer.py gt_%05d.txt > /dev/null 2>&1' % i
                # commands.append(command)
                command = self.blender_path + ' --background --python /home/crl-5/Desktop/cloth_recon/code/src/utils/self_intersec.py %spred_%05d.txt > /dev/null 2>&1' % (basedir, i)
                commands.append(command)

            with Pool(4) as pool:
                pool.map(os.system, commands)

            # calculate self intersection using obj file
            intersec_loss = 0.
            for i in range(gt_vertices.shape[0]):
                input_mesh = trimesh.load(basedir+'pred_%05d.txt.obj' % i)
                vtx = input_mesh.vertices
                face = input_mesh.faces.astype(np.int64)
                triangles = vtx[face]
                triangles = torch.tensor(triangles, dtype=torch.float32, device=device).unsqueeze(dim=0)
                torch.cuda.synchronize()
                outputs = self.m(triangles)
                pdb.set_trace()
                outputs = outputs.detach().cpu().numpy().squeeze()
                collisions = outputs[outputs[:, 0] >= 0, :]
                ratio = collisions.shape[0] / float(triangles.shape[1])
                intersec_loss += ratio
                torch.cuda.synchronize()

            # gt_imgs = []
            # pred_imgs = []
            # for i in range(gt_vertices.shape[0]):
            #     gt_img = cv2.imread('/home/crl-5/Desktop/cloth_recon/tmp/gt_%05d.png0.png'%i, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
            #     pred_img = cv2.imread('/home/crl-5/Desktop/cloth_recon/tmp/pred_%05d.png0.png'%i, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
            #     gt_imgs.append(gt_img)
            #     pred_imgs.append(pred_img)

            # gt_images = torch.from_numpy(np.array(gt_imgs)).float()
            # pred_images = torch.from_numpy(np.array(pred_imgs)).float()
            # for iii, bbb in enumerate(self.data_loader_image_loss):
            #     pred_images = bbb['pred_img']
            #     gt_images = bbb['gt_img']

            shutil.rmtree(basedir)
            '''

            if preds_storage is not None:
                if not self.read_intermediate:
                    preds_storage.append(pred_vertices)
                    if self.attention:
                        if it == 0:
                            attention_storage = [[] for i in range(len(attention_list))]
                        for i in range(len(attention_list)):
                            attention_storage[i].append(attention_list[i])
                else:
                    if it == 0:
                        preds_storage = [[] for i in range(len(output))]
                    for i in range(len(output)):
                        preds_storage[i].append(output[i])

            # compute 3d vertex loss
            loss_vertices = self.vertices_loss(pred_vertices, gt_vertices)

            # add regularizer loss: bernhard
            # edge_len = torch.sqrt(torch.square(torch.index_select(pred_vertices, 1, self.edge_idx[:,0]) - torch.index_select(pred_vertices, 1, self.edge_idx[:,1])).sum(axis=-1))
            # loss_reg = self.reg_loss_cri(edge_len, torch.full(edge_len.shape, 0.04).cuda())
            # compute l1 image loss
            # loss_image = self.image_loss(pred_images, gt_images)
            # loss = loss_vertices + loss_image * 10.

            # add self intersection loss to prevent self intersection
            # loss_intersec = torch.tensor(intersec_loss / batch['image'].shape[0])

            # loss = loss_vertices + loss_intersec * 10
            loss = loss_vertices
            batch_size = batch['image'].shape[0]
            self.log_loss_vertices.update(loss_vertices.item(), batch_size)
            # self.log_loss_image.update(loss_image.item(), batch_size)
            self.log_losses.update(loss.item(), batch_size)
            # self.log_loss_reg.update(loss_reg.item(), batch_size)
            # self.log_loss_intersec.update(loss_intersec.item(), batch_size)

            if loss.requires_grad:
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
                str_print += f"V loss: {self.log_loss_vertices.avg:.10f}  "
                # str_print += f"reg loss: {self.log_loss_reg.avg:.10f}  "
                # str_print += f"image loss: {self.log_loss_image.avg:.10f}  "
                # str_print += f"intersec: {self.log_loss_intersec.avg:.10f}  "
                str_print += f"lr: {opt.param_groups[0]['lr']:.10f}  "
                self.logger.info(str_print)


        if epoch is not None and it == len(data_loader)-1:
            self.logger.info(f"***** Epoch {epoch:03d} mean loss: {self.log_losses.avg:.10f} *****\t")

        if self.attention:
            return self.log_losses.avg, preds_storage, attention_storage
        if preds_storage is not None:
            return self.log_losses.avg, preds_storage
        return self.log_losses.avg # , preds_storage, preds_storage_reg


    def adjust_learning_rate(self, opt, epoch):
        lr = 0.0001 * (0.1 ** (epoch // self.scheduler_step))
        for param_group in opt.param_groups:
            param_group['lr'] = lr


    def send_to_device(self, batch):
        for k, v in batch.items():
            batch[k] = v.to(self.dev)


    def train_model(self, pt_file=''):

        loss = float('inf')
        epoch_start = 0
        best_model_state = {}
        best_loss = float('inf')
        best_epoch = -1
        if pt_file != '':
            checkpoint = torch.load(pt_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            print('>>>>>>>>>>>>>>> load state dict successfully <<<<<<<<<<<<<<<')
        for e in range(epoch_start+1 if epoch_start != 0 else 0, self.num_train_epochs):
            self.logger.info(f"\nEpoch: {e+1:04d}/{self.num_train_epochs:04d}")
            # Train one epoch
            self.model.train()
            self.logger.info("##### TRAINING #####")
            loss_tot = self.one_pass(self.data_loader_train, phase="train", epoch=e)
            # Evaluate on validation set
            with torch.no_grad():
                self.model.eval()
                self.logger.info("##### EVALUATION #####")
                loss_tot = self.one_pass(self.data_loader_val, phase="eval", epoch=e)
                if loss_tot < loss:
                    loss = loss_tot
                    best_loss = loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = e
                    # if not os.path.exists(os.path.join(self.exp_dir, f"bestmodel_{e:04d}_{best_loss:.10f}.pt")):
                    #     torch.save({
                    #         'epoch': e,
                    #         'model_state_dict': best_model_state,
                    #         'optimizer_state_dict': self.opt.state_dict(),
                    #         }, os.path.join(self.exp_dir, f"bestmodel_{e:04d}_{best_loss:.10f}.pt"))

            if (e % self.save_freq) == 0:
                torch.save(
                    {
                        'epoch': e,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                    },
                    os.path.join(self.exp_dir, f"model_{e:04d}_{loss_tot:.10f}.pt"),
                )

        torch.save({
                'model_state_dict': best_model_state,
            }
            , os.path.join(self.exp_dir, f"bestmodel_{best_epoch:04d}_{best_loss:.10f}.pt")
        )


    def test_model(self, pt_file):
        """
        Runs model on testing data
        """
        print("##### TESTING #####")
        checkpoint = torch.load(pt_file) 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            self.model.eval()
            if not self.attention:
                loss_tot, preds_storage = self.one_pass(
                    self.data_loader_test, phase="test", preds_storage=[]
                )
            else:
                loss_tot, preds_storage, attention_storage = self.one_pass(self.data_loader_test, phase='test', preds_storage=[])
  
        print('total loss: ', loss_tot)

        if os.path.exists('/home/crl-5/Desktop/50/pred'):
            shutil.rmtree('/home/crl-5/Desktop/50/pred')
        os.mkdir('/home/crl-5/Desktop/50/pred')

        if not self.read_intermediate:
            preds = pyt2np(torch.cat(preds_storage, dim=0)).tolist()
            for idx, pred in enumerate(preds):
                np.savetxt('/home/crl-5/Desktop/50/pred/%05d_pred.txt'%idx, pred)
            if self.attention:
                for i in range(len(attention_storage)):
                    attentions = pyt2np(torch.cat(attention_storage[i], dim=0)).tolist()
                    for idx, att in enumerate(attentions):
                        np.savetxt('/home/crl-5/Desktop/50/pred/%05d_%02d_attention.txt' % (idx, i), att)
        else:
            for i in range(len(preds_storage)):
                preds = pyt2np(torch.cat(preds_storage[i], dim=0)).tolist()
                for idx, pred in enumerate(preds):
                    np.savetxt('/home/crl-5/Desktop/50/pred/%05d_%02d_pred.txt' % (idx, i), pred)
