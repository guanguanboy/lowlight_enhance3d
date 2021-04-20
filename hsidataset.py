import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from os import listdir
from os.path import join
import scipy.io as scio

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

class HsiTrainDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiTrainDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        lowlight = mat['lowlight'].astype(np.float32)
        label = mat['label'].astype(np.float32)

        #增加一个维度
        lowlight_exp = np.expand_dims(lowlight, axis=0)
        label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(lowlight_exp), torch.from_numpy(label_exp)

    def __len__(self):
        return len(self.image_filenames)

class HsiValDataset(Dataset):
    def __init__(self, dataset_dir):
        super(HsiValDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        lowlight = mat['lowlight'].astype(np.float32).transpose(2, 0, 1)
        label = mat['label'].astype(np.float32).transpose(2, 0, 1)

        #增加一个维度
        lowlight_exp = np.expand_dims(lowlight, axis=0)
        label_exp = np.expand_dims(label, axis=0)

        return torch.from_numpy(lowlight_exp).float(), torch.from_numpy(label_exp).float()

    def __len__(self):
        return len(self.image_filenames)

def run_dataset_test():
    batch_size = 2
    #train_set = HsiTrainDataset('D:\DataSets\hyperspectraldatasets\lowlight_hyperspectral_datasets\lowlight\train')
    train_set = HsiTrainDataset('/mnt/liguanlin/DataSets/hypserdatasets/lowlight/train')
    train_loader = DataLoader(dataset=train_set,  batch_size=batch_size, shuffle=True)  
    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(len(train_loader))

    val_set = HsiValDataset('/mnt/liguanlin/DataSets/hypserdatasets/lowlight/test')
    val_batch_size = 1
    val_loader = DataLoader(dataset=val_set,  batch_size=val_batch_size, shuffle=False)  
    print(next(iter(val_loader))[0].shape)
    print(next(iter(val_loader))[1].shape)
    print(len(val_loader))    


#run_dataset_test()

