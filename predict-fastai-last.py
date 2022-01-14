import torch
import numpy as np
import pandas
import os
import timm
# import gc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
# import wandb
# from datetime import datetime
# import pprint

from utils import set_random_seed, get_criterion, val_epoch, root_mean_square_error
from config import Paths, Training
from augmentations import get_augmentations_val
from model import Model
from dataset import PawpularDataset

from utils import sigmoid_np
from sklearn.metrics import mean_squared_error

DEBUG = False

seed=365
set_random_seed(seed)
device = torch.device('cuda')

Training.use_meta = False

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
#df['fold'] = np.random.randint(low=0, high=Training.n_folds, size=len(df))

num_bins = int(np.ceil(2*((len(df))**(1./3))))
df['bins'] = pandas.cut(df['Pawpularity'], bins=num_bins, labels=False)

df['fold'] = -1

N_FOLDS = 10
strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(df.index, df['bins'])):
    df.iloc[train_index, -1] = i

df['fold'] = df['fold'].astype('int')


oof_predictions = []
targets = np.concatenate([df[df['fold'] == fold]['Pawpularity'].values for fold in range(Training.n_folds)]) / 100.

testing_models = [('swin_large_patch4_window12_384_in22k', 384),  # 1
                  ('swin_large_patch4_window7_224_in22k', 224)    # 3
                 ]

for model_name, image_size in testing_models:
    pred = []
    Training.image_size = image_size

    for fold in range(10):
        val_df = df[df['fold'] == fold]

        val_dataset = PawpularDataset(csv=val_df, data_path=Paths.data,
                                      augmentations=get_augmentations_val(Training), meta_features=meta_features)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=Training.batch_size,
                                                 num_workers=Training.num_workers)

        #model = Model(kernel_type=model_name, n_meta_features=n_meta_features, n_meta_dim=[512, 128], pretrained=False)
        #model_path = os.path.join(Paths.weights, f'{model_name}_fold_{fold}_best.pth')
        model = timm.create_model(model_name, pretrained=False, num_classes=1)
        model_path = f'{model_name}_fold_{fold}.pth'
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        pred.append(val_epoch(model=model, loader=val_loader, criterion=criterion,
                              use_meta=Training.use_meta, device=device, DEBUG=DEBUG, get_output=True))

    oof_predictions.append(np.concatenate(pred))


# TESTING RESULTS
# solo
for i, model_name in enumerate(testing_models):
    print(f'{model_name}: {root_mean_square_error(oof_predictions[i], targets):.5f}')

# concat
stack = np.stack(oof_predictions)
for i in range(len(testing_models)):
    print(f'Model 0-{i}: {root_mean_square_error(np.mean(stack[:i+1], axis=0), targets):.5f};')
    #print(f'Model 0-{i}: {mean_squared_error(np.mean(sigmoid_np(stack[:i + 1]), axis=0) * 100, targets * 100) ** 0.5:.5f}.')