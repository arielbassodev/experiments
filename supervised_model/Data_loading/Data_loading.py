import numpy as np
import torch
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks
from sympy.physics.units import length
from  torch.utils.data import DataLoader, Dataset, TensorDataset

class DataLoading:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def data_preparation(self):
        features = [np.concatenate((items['left_hand'], items['right_hand'], items['pose']), axis=1) for items, _ in self.dataset]
        features = [items.reshape(items.shape[:-2] + (-1,)) for items in features]
        max_frame_length = max(frame.shape[0] for frame in features)
        frame_padded = [np.pad(items, ((0, max_frame_length - items.shape[0]), (0, 0)), mode='constant') for items in features]
        masks = [[True]*items.shape[0] + [False]*(max_frame_length - items.shape[0]) for items in features]
        labels = [label for _, label in self.dataset]
        tensor_frame = torch.Tensor(frame_padded)
        tensor_label = torch.Tensor(labels)
        mask = torch.Tensor(masks)
        return tensor_frame, tensor_label, mask

    def build_dataset_from_tensor(self, feature_tensor, target_tensor, masks):
        dataset = TensorDataset(feature_tensor, target_tensor,masks)
        return dataset

    def build_data_loader(self,dataset):
        train_dataset = DataLoader(dataset, batch_size=512, shuffle=True)
        return train_dataset

    def __call__(self, *args, **kwargs):
        tensor_frame, tensor_label, masks = self.data_preparation()
        data = self.build_dataset_from_tensor(tensor_frame, tensor_label, masks)
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        return dataloader



