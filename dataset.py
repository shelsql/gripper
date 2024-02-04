import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self,
                 dataset_location,
                 use_augs=False
                 ):
        super().__init__()
        print("Loading pose estimation dataset...")
        self.dataset_location = dataset_location
    def __getitem__(self, index):
        return
    def __len__(self):
        return
        

class TrackingDataset(Dataset):
    def __init__(self,
                 dataset_location="/root/autodl-tmp/shiqian/datasets/Ty_data",
                 use_augs=False,
                 S=8,
                 ):
        super().__init__()
        print("Loading pose tracking dataset...")
        self.dataset_location = dataset_location
    def __getitem__(self, index):
        return
    def __len__(self):
        return