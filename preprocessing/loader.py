import os
import torch
from torchvision import transforms, datasets
from augmentation import Aug

#Install TPU environment using this code
#!curl install https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
#!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev

mean = [0.485, 0.456, 0.406] #[0.4718, 0.3467, 0.3154] DFDC dataset mean and standard
std = [0.229, 0.224, 0.225]  #[0.1656, 0.1432, 0.1364]

data_transforms = {
    'train': transforms.Compose([
        Aug(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# load data

def session(cession='g', data_dir = 'sample/', batch_size=32):
    batch_size=batch_size
    data_dir = data_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}

    if cession=='t':
        dataloaders, dataset_sizes = load_tpu(image_datasets, batch_size, data_dir)
        return batch_size, dataloaders, dataset_sizes
    else:
        dataloaders, dataset_sizes = load_gpu(image_datasets, batch_size, data_dir)
        return batch_size, dataloaders, dataset_sizes

def load_gpu(image_datasets, batch_size, data_dir):

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                                 shuffle=True, num_workers=0, pin_memory=True)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes

def load_tpu(batch_size, data_dir):

    # imports the torch_xla package
    import torch_xla
    import torch_xla.core.xla_model as xm

    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.utils.utils as xu

    # LOAD TPU

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    train_sampler = {x: torch.utils.data.distributed.DistributedSampler(
    image_datasets[x],
    num_replicas=xm.xrt_world_size(),
    rank=xm.get_ordinal(),
    shuffle=True) for x in ['train', 'val', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(
      image_datasets[x],
      batch_size,
      sampler=train_sampler[x],
      num_workers=0,
      drop_last=True,
      pin_memory=True) for x in ['train', 'val', 'test']}

    # Scale learning rate to world size
    lr = 0.0001 * xm.xrt_world_size()

    return dataloaders, dataset_sizes

def curriculum_loader(cession, dir_path, batch_size, difficulty_level):
    # Define transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            Aug(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Modify the data directory to include the difficulty level
    data_dir = os.path.join(dir_path, 'train', difficulty_level)  # for training data
    val_dir = os.path.join(dir_path, 'val', difficulty_level)      # for validation data
    test_dir = os.path.join(dir_path, 'test', difficulty_level)    # for test data

    image_datasets = {
        'train': datasets.ImageFolder(data_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }

    if cession == 't':
        return load_tpu(batch_size, data_dir, image_datasets)
    else:
        return load_gpu(image_datasets, batch_size, data_dir)
