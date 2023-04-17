from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
import random
import numpy as np


class cifarDataset(Dataset):
    def __init__(self, train=True,):
        super(cifarDataset, self).__init__()
        self.train = train
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                                         std=np.array([63.0, 62.1, 66.7]) / 255.0)])
        self.dataset = datasets.cifar.CIFAR100(root='data', train=train, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=np.array([125.3, 123.0, 113.9]) / 255.0,
                                                         std=np.array([63.0, 62.1, 66.7]) / 255.0),
             ]), download=False)
        targets = np.array(self.dataset.targets)
        self.idxs = []
        for i in range(100):
            idx = np.where(targets == i)
            self.idxs.append(idx)
        self.task_list = [i for i in range(100)]


    def __getitem__(self, idx):
        random.shuffle(self.task_list)
        self.task = [self.task_list[i] for i in range(10)]  # random.choice(self.task_list)
        prompt = torch.zeros(100).scatter_(0, torch.tensor(self.task), 1)

        label = random.choice(self.task)

        image_list = self.idxs[label][0].tolist()

        image_idx = random.choice(image_list)

        image = self.dataset.data[int(image_idx)]
        image = self.transform(image)

        return prompt, image, int(label)

    def __len__(self):

        if self.train:
            return 100000
        else:
            return 20000


