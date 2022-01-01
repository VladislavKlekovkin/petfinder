#!pip install timm
import timm
import random
import torch
import numpy as np
import pandas
import os
import gc
from tqdm import tqdm
import cv2
import albumentations as A
from sklearn.metrics import mean_squared_error
import wandb
from datetime import datetime
import pprint


DEBUG = True


# REPRODUCIBILITY
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_random_seed()
device = torch.device('cuda')


class AttributeInspect:
    
    @classmethod
    def get_class_attributes(cls):
        attributes = {}
        for attr in cls.__dict__.keys():
            if attr[:2] != '__':
                value = getattr(cls, attr)
                if not callable(value):
                    attributes[attr] = value
        
        return attributes

    
class Paths(AttributeInspect):
    
    inp = '../input'
    outp = '../output'
    data = os.path.join(inp, 'petfinder-pawpularity-score')
    train_csv = os.path.join(data, 'train.csv')
    test_csv = os.path.join(data, 'test.csv')
    sample_submission = os.path.join(data, 'sample_submission.csv')
    weights = os.path.join(inp, 'weights')

        
class Training(AttributeInspect):
    
    kernel_type = 'swin_large_patch4_window7_224'  # 'efficientnetv2_m', 'swin_large_patch4_window7_224'
    epochs = 7
    warm_up_epochs = 1
    n_folds = 5
    batch_size = 2
    patience = float('inf')
    num_workers = 4
    image_size = 224
    use_meta = True
    lr = 5e-5
    scheduler = 'LambdaLR'
    if not use_meta:
        warm_up_epochs = 0
    if DEBUG:
        epochs = 2
        n_folds = 2
    
    
def sigmoid_np(x):
  
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def root_mean_square_error(predictions, targets):
    
    return mean_squared_error(sigmoid_np(predictions) * 100., targets * 100.) ** .5


def get_scheduler(scheduler, optimizer):
    
    if scheduler is None:
        return None
    
    if scheduler is 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Training.lr,
                                                   steps_per_epoch=1, epochs=Training.epochs)
    
    if scheduler is 'LambdaLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: .1 ** e)


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

augmentations_train = A.Compose([
    # A.Transpose(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.HorizontalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.RandomBrightness(limit=0.2, p=0.75),
    # A.RandomContrast(limit=0.2, p=0.75),
    # A.OneOf([
    #     A.MotionBlur(blur_limit=5),
    #     A.MedianBlur(blur_limit=5),
    #     A.GaussianBlur(blur_limit=5),
    #     A.GaussNoise(var_limit=(5.0, 30.0)),
    # ], p=0.7),
    #
    # A.OneOf([
    #     A.OpticalDistortion(distort_limit=1.0),
    #     A.GridDistortion(num_steps=5, distort_limit=1.),
    #     A.ElasticTransform(alpha=3),
    # ], p=0.7),
    #
    # A.CLAHE(),
    # A.HueSaturationValue(),
    # A.RandomBrightness(),
    #
    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(Training.image_size, Training.image_size),
    # A.Cutout(max_h_size=int(cfg.Training.image_size * 0.375),
    #          max_w_size=int(cfg.Training.image_size * 0.375), num_holes=1, p=0.7),
    A.Normalize()
])

augmentations_val = A.Compose([
    A.Resize(Training.image_size, Training.image_size),
    A.Normalize()
])

sigmoid_torch = torch.nn.Sigmoid()


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid_torch(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = sigmoid_torch(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SwishModule(torch.nn.Module):

    def forward(self, x):
        return Swish.apply(x)
        

class Model(torch.nn.Module):
    
    def __init__(self, kernel_type, n_meta_features=0, n_meta_dim=[512, 128]):
        
        super().__init__()
        self.n_meta_features = n_meta_features
        self.model = timm.create_model(kernel_type, pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.dropouts = torch.nn.ModuleList([
            torch.nn.Dropout(0.5) for _ in range(5)
        ])

        if 'swin_' in kernel_type:
            in_features = self.model.head.in_features
            self.model.head = torch.nn.Identity()
        elif 'efficientnetv2_' in kernel_type:
            in_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Identity()
            
        if n_meta_features > 0:
            self.meta = torch.nn.Sequential(
                torch.nn.Linear(n_meta_features, n_meta_dim[0]),
                torch.nn.BatchNorm1d(n_meta_dim[0]),
                SwishModule(),
                torch.nn.Dropout(p=0.3),
                torch.nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                torch.nn.BatchNorm1d(n_meta_dim[1]),
                SwishModule(),
            )
            in_features += n_meta_dim[1]
            
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x, x_meta=None):
        
        x = self.model(x)
        
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
            
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.linear(dropout(x))
            else:
                out += self.linear(dropout(x))
                
        out /= len(self.dropouts)        
        
        return out

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


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
    n_meta_features = 0

# Adding bins to train.csv
df['fold'] = np.random.randint(low=0, high=Training.n_folds, size=len(df))


class PawpularDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv, mode='train', augmentations=None):
        self.csv = csv
        self.mode = mode
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image_path = os.path.join(Paths.data, self.mode, f'{row["Id"]}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
                                  
        if Training.use_meta:
            data = (torch.tensor(image, dtype=torch.float), 
                    torch.tensor(row[meta_features], dtype=torch.float))
        else:                       
            data = torch.tensor(image, dtype=torch.float)
        
        if self.mode == 'test':
            return data
        
        return data, torch.tensor([row['Pawpularity'] / 100.], dtype=torch.float)
        
        
criterion = torch.nn.BCEWithLogitsLoss()


def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    PREDICTIONS = []
    TARGETS = []
    bar = tqdm(loader)
    for (data, target) in bar:
        optimizer.zero_grad()
        if Training.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            predictions = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            predictions = model(data)
        
        PREDICTIONS.append(predictions.detach().cpu())
        TARGETS.append(target.detach().cpu())
        
        loss = criterion(predictions, target)

        loss.backward()
        
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
        if DEBUG:
            break
            
    PREDICTIONS = torch.cat(PREDICTIONS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    rmse = root_mean_square_error(PREDICTIONS, TARGETS)
    
    return train_loss, rmse


def val_epoch(model, loader, get_output=False):
    
    model.eval()
    val_loss = []
    PREDICTIONS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            if Training.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                predictions = model(data, meta)
            else:
                data, target = data.to(device), target.to(device)
                predictions = model(data)

            PREDICTIONS.append(predictions.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(predictions, target)
            val_loss.append(loss.detach().cpu().numpy())
            if DEBUG:
                break

    val_loss = np.mean(val_loss)
    PREDICTIONS = torch.cat(PREDICTIONS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return PREDICTIONS
    else:
        rmse = root_mean_square_error(PREDICTIONS, TARGETS)
        return val_loss, rmse


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
        train_dataset = PawpularDataset(csv=train_df, augmentations=augmentations_train)
        val_dataset = PawpularDataset(csv=val_df, augmentations=augmentations_val)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=Training.batch_size,
                                                   sampler=torch.utils.data.sampler.RandomSampler(train_dataset), 
                                                   num_workers=Training.num_workers,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=Training.batch_size,
                                                 num_workers=Training.num_workers)
        
        model = Model(kernel_type=Training.kernel_type, n_meta_features=n_meta_features)
        model = model.to(device)

        patience_counter = 0
        val_rmse_min = float('inf')
        model_file = os.path.join(Paths.weights, f'{model_name}_best.pth')

        optimizer = torch.optim.Adam(model.parameters(), lr=Training.lr)
        scheduler = get_scheduler(Training.scheduler, optimizer)

        for epoch in range(1, Training.epochs + 1):
            print('Epoch:', epoch)
            if (epoch - 1) == Training.warm_up_epochs:
                model.unfreeze()
            train_loss, train_rmse = train_epoch(model, train_loader, optimizer)
            val_loss, val_rmse = val_epoch(model, val_loader)
            
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
setattr(Training, 'kernel_type', 'swin_large_patch4_window7_224_in22k')
run(notes='Changed model: swin_large_patch4_window7_224_in22k')

setattr(Training, 'image_size', 384)

setattr(Training, 'kernel_type', 'swin_large_patch4_window12_384')
run(notes='Changed model: swin_large_patch4_window12_384')

setattr(Training, 'kernel_type', 'swin_large_patch4_window12_384_in22k')
run(notes='Changed model: swin_large_patch4_window12_384_in22k')

# for value in experiments['values']:
#     setattr(Training, experiments['parameter'], value)
#     notes = f"Set {experiments['parameter']} to {value}"
#     run(notes=notes)
