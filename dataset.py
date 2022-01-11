import torch
import numpy as np
import cv2
import os


class PawpularDataset(torch.utils.data.Dataset):

    def __init__(self, csv, data_path, mode='train', augmentations=None, meta_features=None):
        self.csv = csv
        self.data_path = data_path
        self.mode = mode
        self.augmentations = augmentations
        self.meta_features = meta_features

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image_path = os.path.join(self.data_path, self.mode, f'{row["Id"]}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.meta_features:
            data = (torch.tensor(image, dtype=torch.float),
                    torch.tensor(row[self.meta_features], dtype=torch.float))
        else:
            data = torch.tensor(image, dtype=torch.float)

        # if self.mode == 'test':
        #     return data

        return data, torch.tensor([row['Pawpularity'] / 100.], dtype=torch.float)
