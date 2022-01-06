import albumentations as A


def get_augmentations_train(Training):
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

    return augmentations_train


def get_augmentations_val(Training):
    augmentations_val = A.Compose([
        A.Resize(Training.image_size, Training.image_size),
        A.Normalize()
    ])

    return augmentations_val
