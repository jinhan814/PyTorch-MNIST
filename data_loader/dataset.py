from torch.utils.data import Dataset
import torch
import json
import os
import cv2


class MnistDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path      = path
        self.info      = json.load(open(os.path.join(path, 'info.json'), 'r'))
        self.transform = transform

    def __len__(self, ):
        return len(self.info)

    def __getitem__(self, idx):
        img_path = self.info[str(idx)]['img_path']
        img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255
        img      = torch.tensor(img, dtype=torch.float32)
        label    = int(self.info[str(idx)]['label'])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
