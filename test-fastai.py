from timm import create_model
from fastai.vision.all import *
import math
import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import mean_squared_error

def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))

def sigmoid_np(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def root_mean_square_error(predictions, targets):
    return mean_squared_error(sigmoid_np(predictions) * 100., targets * 100.) ** .5


seed=365
set_seed(seed, reproducible=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

BATCH_SIZE = 16

dataset_path = Path('../input/petfinder-pawpularity-score/')
train_df = pd.read_csv(dataset_path/'train.csv')

train_df['path'] = train_df['Id'].map(lambda x:str(dataset_path/'train'/x)+'.jpg')
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe
train_df['norm_score'] = train_df['Pawpularity'] / 100

num_bins = int(np.ceil(2*((len(train_df))**(1./3))))
train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)

train_df['fold'] = -1

N_FOLDS = 10
strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=seed, shuffle=True)
for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
    train_df.iloc[train_index, -1] = i

train_df['fold'] = train_df['fold'].astype('int')

oof_predictions = []
targets = np.concatenate([train_df[train_df['fold'] == fold]['Pawpularity'].values for fold in range(N_FOLDS)]) / 100.

testing_models = [('swin_large_patch4_window12_384', 384)  # 1
                  # ('swin_large_patch4_window12_384', 384),        # 2
                  # ('swin_large_patch4_window7_224_in22k', 224),   # 3
                  # ('swin_large_patch4_window7_224', 224)          # 4
                 ]
# train_df = train_df.head(100)
# targets = targets[:100]
for kernel_type, img_size in testing_models:

    pred =[]
    dls = ImageDataLoaders.from_df(train_df,  # pass in train DataFrame
                                   valid_pct=0.2,  # 80-20 train-validation random split
                                   seed=365,  # seed
                                   fn_col='path',  # filename/path is in the second column of the DataFrame
                                   label_col='norm_score',  # label is in the first column of the DataFrame
                                   y_block=RegressionBlock,  # The type of target
                                   bs=BATCH_SIZE,  # pass in batch size
                                   num_workers=8,
                                   item_tfms=Resize(img_size),  # pass in item_tfms
                                   batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()]))

    for fold in range(N_FOLDS):
        val_df = train_df[train_df['fold'] == fold]

        model = create_model(kernel_type, pretrained=False, num_classes=dls.c)
        #model.load_state_dict(torch.load(f'{kernel_type}_fold_{i}.pth'))
        learn = Learner(dls, model, loss_func=BCEWithLogitsLossFlat(), model_dir='').to_fp16()
        learn.load(f'{kernel_type}_fold_{fold}')

        val_dl = dls.test_dl(val_df)
        preds, _ = learn.tta(dl=val_dl, n=1, beta=0)
        pred.append(preds)

        del learn
        torch.cuda.empty_cache()
        gc.collect()

    oof_predictions.append(np.concatenate(pred))

for i, model_name in enumerate(testing_models):
    print(f'{model_name}: {root_mean_square_error(oof_predictions[i], targets):.5f}')

stack = np.stack(oof_predictions)
for i in range(len(testing_models)):
    print(f'Model 0-{i}: {root_mean_square_error(np.mean(stack[:i+1], axis=0), targets):.5f};')

