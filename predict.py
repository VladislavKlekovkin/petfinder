import torch
import numpy as np
import pandas
import os
# import gc
from tqdm import tqdm
# import wandb
# from datetime import datetime
# import pprint

from utils import set_random_seed, get_criterion, val_epoch, root_mean_square_error
from config import Paths, Training
from augmentations import get_augmentations_val
from model import Model
from dataset import PawpularDataset

DEBUG = False

set_random_seed()
device = torch.device('cuda')

criterion = get_criterion(Training)

# Reading data
df = pandas.read_csv(Paths.train_csv)

# Meta features
if Training.use_meta:

    # Adding image sizes
    image_names = df['Id'].values
    image_sizes = np.zeros(image_names.shape[0])
    for i, img_name in enumerate(tqdm(image_names)):
        image_sizes[i] = os.path.getsize(os.path.join(Paths.data, 'train', f'{img_name}.jpg'))
    df['Image_size'] = np.log(image_sizes)

    meta_features = list(filter(lambda e: e not in ('Id', 'fold', 'Pawpularity'), df.columns))
    n_meta_features = len(meta_features)
else:
    meta_features = None
    n_meta_features = 0

# Adding bins to train.csv
df['fold'] = np.random.randint(low=0, high=Training.n_folds, size=len(df))

oof_predictions = []
targets = df['Pawpularity'].values

testing_models = [('swin_large_patch4_window12_384_in22k_02-01-2022-23:53:35', 384),  # 1
                  # ('swin_large_patch4_window12_384_02-01-2022-10:46:15', 384),        # 2
                  # ('swin_large_patch4_window7_224_in22k_02-01-2022-04:51:06', 224),   # 3
                  # ('swin_large_patch4_window7_224_29-12-2021-21:45:26', 224)          # 4
                 ]

for model_name, image_size in testing_models:
    pred = []
    Training.image_size = image_size

    for fold in range(Training.n_folds):
        val_df = df[df['fold'] == fold]

        val_dataset = PawpularDataset(csv=val_df, data_path=Paths.data,
                                      augmentations=get_augmentations_val(Training), meta_features=meta_features)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=Training.batch_size,
                                                 num_workers=Training.num_workers)

        model = Model(kernel_type=model_name[:-20], n_meta_features=n_meta_features)
        model_path = os.path.join(Paths.weights, f'{model_name}_fold_{fold}_best.pth')
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        # pred.append(val_epoch(model=model, loader=val_loader, criterion=criterion,
        #                       use_meta=Training.use_meta, device=device, DEBUG=DEBUG, get_output=True))
        val_loss, val_rmse = val_epoch(model=model, loader=val_loader, criterion=criterion,
                                       use_meta=Training.use_meta, device=device, DEBUG=DEBUG)
        print(val_rmse)

    # oof_predictions.append(np.concatenate(pred))


# TESTING RESULTS
# solo
# for i, model_name in enumerate(testing_models):
#     print(f'{model_name}: {root_mean_square_error(oof_predictions[i], targets):.5f}')
