import argparse
import os
import datetime

# This lib allows accessing dict keys via `.`. E.g d['item'] == d.item
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import shutil

from src.models import model_factory
from src.dataset import transforms_factory, data_factory
from src.trainer import Trainer
from src.utils.utils import worker_init, set_seed
from src.utils.utils import add_logging

main_seed = 0
set_seed(main_seed)  # Seed main thread
num_threads = 8
# all_data
# data_dir = '/home/crl-5/Desktop/dataset'
data_dir = '/home/crl-5/Desktop/50'
save_dir = '/home/crl-5/Desktop/cloth_recon/experiments'
######## Set-up experimental directories
if os.path.exists('/home/crl-5/Desktop/cloth_recon/tmp'):
    shutil.rmtree('/home/crl-5/Desktop/cloth_recon/tmp')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend", help="backend network", type=str, default="resnet34"
)
parser.add_argument("--lr", help="learning rate of optimizer", type=float, default=1e-4)
parser.add_argument(
    "--n_epochs", help="Number of training epochs", type=int, default=200
)
parser.add_argument(
    "--batch_size", help="Batch size for one pass", type=int, default=8
)
parser.add_argument(
    "--dev", help="Use cpu or cuda", type=str, default="cuda"
)
parser.add_argument(
    "--phase", help="train/test", type=str, default="train"
)
parser.add_argument(
    "--pt_file", type=str, default=''
)
parser.add_argument(
    '--scheduler_step', type=int, default=100
)
parser.add_argument(
    '--use_pretrained', type=bool, default=False
)
parser.add_argument(
    '--read_intermediate', type=bool, default=False
)
parser.add_argument(
    '--show_attention', type=bool, default=False
)
args = parser.parse_args()


pt_file = args.pt_file
pt_name = ''
if args.phase == 'train':
    unique_id = str(datetime.datetime.now().microsecond)
    exp_dir = os.path.join(save_dir, f"exp_{unique_id}_{args.backend}_mlp_baseline")
    # If this fails, there was an ID clash. Hence its preferable to crash than overwrite
    os.mkdir(exp_dir)
    # file = open(os.path.join(exp_dir, 'out.txt'), 'w+')
    # sys.stdout = file
else:
    exp_name = args.pt_file.split('/')[-2]
    pt_name = args.pt_file.split('/')[-1][:-3]
    exp_dir = os.path.join(save_dir, exp_name)

logger = add_logging("clothrecon", os.path.join(exp_dir,'log.txt'))

######### Set-up model
model_cfg = edict(
    {
        "name": "main_model",
        "backend": {
            "name": args.backend,  # Defines the backend model type
            'use_pretrained': args.use_pretrained,
            "output_slices": {
                "verts": 2601*3 #deprecated
            },  # Defines the outputs and their dimensionality
        },
        "cloth_model": {
            "name": "cloth_model",
        },
        # "graph_cnn": {
        #     "name": "graph_cnn",
        # }
    }
)
dev = torch.device(args.dev)
model = model_factory.get_model(model_cfg, dev, logger)
######### Set-up optimizer
opt = optim.Adam(model.parameters(), lr=args.lr)
######### Set-up data transformation
transformation_cfg = edict(
    {
        # TODO!
        # "Augmentation": {},
        "Resize": {"img_size": (224, 224)},
        "NumpyToPytorch": {},
        # "Normalize": {}
    }
)
transformations = transforms_factory.get_transforms(transformation_cfg)

transformation_cfg_eval = edict(
    {
        "Resize": {"img_size": (224, 224)},
        "NumpyToPytorch": {},
        # "Normalize": {}
    }
)
# transformation_cfg_il = edict({'NpToPytSingleChannel': {}})
transformations_eval = transforms_factory.get_transforms(transformation_cfg_eval)
# transformations_il = transforms_factory.get_transforms(transformation_cfg_il)

######### Set-up data reader and data loader
data_cfg = edict({"ClothRecon": {"dataset_path": data_dir}})
data_reader_train = data_factory.get_data_reader(
    data_cfg, split="train", data_transforms=transformations
)
data_reader_val = data_factory.get_data_reader(
    data_cfg, split="val", data_transforms=transformations_eval
)
data_reader_test = data_factory.get_data_reader(
    data_cfg, split='test', data_transforms=transformations_eval
)
# data_reader_image_loss = data_factory.get_data_reader_image_loss(args.batch_size, transformations_il)

data_loader_train = DataLoader(
    data_reader_train,
    batch_size=args.batch_size,
    shuffle=True,  # Re-shuffle data at every epoch
    num_workers=num_threads,  # Number of worker threads batching data
    drop_last=True,  # If last batch not of size batch_size, drop
    pin_memory=True,  # Faster data transfer to GPU
    worker_init_fn=lambda x: worker_init(
        x, main_seed
    ),  # Seed all workers. Important for reproducibility
)
data_loader_val = DataLoader(
    data_reader_val,
    batch_size=args.batch_size,
    shuffle=False,  # Go through the test data sequentially. Easier to plot same samples to observe them over time
    num_workers=num_threads,  # Number of worker threads batching data
    drop_last=False,  # We want to validate on ALL data
    pin_memory=True,  # Faster data transfer to GPU
    worker_init_fn=lambda x: worker_init(x, main_seed),
)
data_loader_test = DataLoader(
    data_reader_test,
    batch_size=args.batch_size,
    shuffle=False,  # Go through the test data sequentially. Easier to plot same samples to observe them over time
    num_workers=num_threads,  # Number of worker threads batching data
    drop_last=False,  # We want to validate on ALL data
    pin_memory=True,  # Faster data transfer to GPU
    worker_init_fn=lambda x: worker_init(x, main_seed),
)
# data_loader_image_loss = DataLoader(
#     data_reader_image_loss,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=num_threads,
#     drop_last=False,
#     pin_memory=True,
#     worker_init_fn=lambda x: worker_init(x, main_seed),
# )

######### Set-up trainer and run training
trainer = Trainer(
    model,
    opt,
    data_loader_train,
    data_loader_val,
    data_loader_test,
    # data_loader_image_loss,
    exp_dir,
    dev,
    logger,
    args.n_epochs,
    args.scheduler_step,
    args.read_intermediate,
    args.show_attention
)
if args.phase == 'train':
    trainer.train_model(pt_file)
    # pt_file = os.path.join(exp_dir, 'model_last.pt')
    # trainer.test_model(pt_file)

if args.phase == 'test':
    if pt_file == '':
        import pdb
        pdb.set_trace()
    trainer.test_model(pt_file)

