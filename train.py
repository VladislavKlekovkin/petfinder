#!pip install timm
import torch
import numpy as np
import pandas
import os
import gc
from tqdm import tqdm
import wandb
from datetime import datetime
import pprint

from utils import set_random_seed, get_scheduler, get_criterion, train_epoch, val_epoch
from config import Paths, Training
from augmentations import get_augmentations_train, get_augmentations_val
from model import Model
from dataset import PawpularDataset

DEBUG = True

if DEBUG:
    Training.epochs = 2
    Training.n_folds = 2

set_random_seed()
device = torch.device('cuda')


# def get_trans(img, I):
#     if I >= 4:
#         img = img.transpose(2,3)
#     if I % 4 == 0:
#         return img
#     elif I % 4 == 1:
#         return img.flip(2)
#     elif I % 4 == 2:
#         return img.flip(3)
#     elif I % 4 == 3:
#         return img.flip(2).flip(3)


# model = timm.create_model('efficientnetv2_m')
# print(model.__str__()[-1000:])
# model


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


criterion = get_criterion(Training)


def run(notes='Baseline'):
    
    datetime_suffix = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    group = Training.kernel_type + '_' + datetime_suffix
    
    pprint.pprint(Training.get_class_attributes())
    print(group)

    for fold in range(Training.n_folds):
        
        model_name = group + f'_fold_{fold}'
        if not DEBUG:
            wandb.init(
                project="petfinder",
                entity='vladislav',
                group=group,
                name=model_name,
                notes=notes,
                config=Training.get_class_attributes(),
            )

        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        train_dataset = PawpularDataset(csv=train_df, data_path=Paths.data,
                                        augmentations=get_augmentations_train(Training), meta_features=meta_features)
        val_dataset = PawpularDataset(csv=val_df, data_path=Paths.data,
                                      augmentations=get_augmentations_val(Training), meta_features=meta_features)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=Training.batch_size,
                                                   sampler=torch.utils.data.sampler.RandomSampler(train_dataset), 
                                                   num_workers=Training.num_workers,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=Training.batch_size,
                                                 num_workers=Training.num_workers)
        
        model = Model(kernel_type=Training.kernel_type, n_meta_features=n_meta_features,
                      drop_rate=Training.drop_rate, drop_path_rate=Training.drop_path_rate)
        model = model.to(device)

        patience_counter = 0
        val_rmse_min = float('inf')
        model_file = os.path.join(Paths.weights, f'{model_name}_best.pth')

        optimizer = torch.optim.Adam(model.parameters(), lr=Training.lr)
        scheduler = get_scheduler(Training, optimizer)

        for epoch in range(1, Training.epochs + 1):
            print('Epoch:', epoch)
            if (epoch - 1) == Training.warm_up_epochs:
                model.unfreeze()
            train_loss, train_rmse = train_epoch(model=model, loader=train_loader, optimizer=optimizer,
                                                 criterion=criterion, use_meta=Training.use_meta, device=device,
                                                 DEBUG=DEBUG)
            val_loss, val_rmse = val_epoch(model=model, loader=val_loader, criterion=criterion,
                                           use_meta=Training.use_meta, device=device, DEBUG=DEBUG)
            
            if not DEBUG:
                wandb.log({'train_loss': np.mean(train_loss), 'train_rmse': train_rmse,
                           'val_loss': val_loss, 'val_rmse': val_rmse})
            content = f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, \
                    \ntrain loss: {np.mean(train_loss):.5f}, train rmse: {train_rmse:.4f}, \
                    valid loss: {val_loss:.5f}, val_rmse: {val_rmse:.4f}.'
            print(content)

            if val_rmse < val_rmse_min:
                print('val_rmse_min ({:.6f} --> {:.6f}). Saving model ...'.format(val_rmse_min, val_rmse))
                torch.save(model.state_dict(), model_file)
                if DEBUG:
                    os.remove(model_file)
                else:
                    wandb.run.summary["val_rmse_min"] = val_rmse
                val_rmse_min = val_rmse
                patience_counter = 0

            if patience_counter >= Training.patience:
                print(f"Early stopping at epoch # {epoch}")
                break

            patience_counter += 1
            if scheduler:
                scheduler.step()
        
        # Memory cleaning
        model = None
        optimizer = None
        scheduler = None
        gc.collect()
        torch.cuda.empty_cache()
    
        if not DEBUG:
            wandb.finish()
        
        
# experiments = {
#     'parameter': 'lr',
#     'values': [1e-5, 5e-6, 2e-6, 1e-6]
# }

# Experiments
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnet_b4')
setattr(Training, 'drop_rate', 0.4)
setattr(Training, 'drop_path_rate', 0.2)
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnet_b6')
setattr(Training, 'drop_rate', 0.5)
setattr(Training, 'drop_path_rate', 0.2)
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnet_b7')
setattr(Training, 'drop_rate', 0.5)
setattr(Training, 'drop_path_rate', 0.2)
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnet_b8')
setattr(Training, 'drop_rate', 0.5)
setattr(Training, 'drop_path_rate', 0.2)
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnetv2_m')
setattr(Training, 'drop_rate', 0.)
setattr(Training, 'drop_path_rate', 0.)
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnetv2_l')
setattr(Training, 'drop_rate', 0.)
setattr(Training, 'drop_path_rate', 0.)
run(notes=f"Experiment with {Training.kernel_type}")

setattr(Training, 'kernel_type', 'efficientnetv2_xl')
setattr(Training, 'drop_rate', 0.)
setattr(Training, 'drop_path_rate', 0.)
run(notes=f"Experiment with {Training.kernel_type}")


# setattr(Training, 'kernel_type', 'swin_large_patch4_window7_224_in22k')
# run(notes='Changed model: swin_large_patch4_window7_224_in22k')
#
# setattr(Training, 'image_size', 384)

# setattr(Training, 'kernel_type', 'swin_large_patch4_window12_384')
# run(notes='Changed model: swin_large_patch4_window12_384')
#
# setattr(Training, 'kernel_type', 'swin_large_patch4_window12_384_in22k')
# run(notes='Changed model: swin_large_patch4_window12_384_in22k')

# for value in experiments['values']:
#     setattr(Training, experiments['parameter'], value)
#     notes = f"Set {experiments['parameter']} to {value}"
#     run(notes=notes)
