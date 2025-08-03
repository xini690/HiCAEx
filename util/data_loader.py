import ast
from types import SimpleNamespace

import pandas as pd
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, create_transform, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms
# from util.datasets import build_dataset
def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class EyeDiseaseDataset(Dataset):
    def __init__(self, args,csv_path,transform=None):
        self.csv_path = csv_path
        self.data=pd.read_excel(self.csv_path)
        self.images_number=args.images_number
        self.transform = transform
        self.multi_label_classification=args.multi_label_classification
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.multi_label_classification:
            label = torch.tensor(ast.literal_eval(row['label']), dtype=torch.float32) # 多标签分类用 float32 类型
        else:
            label = torch.tensor(int(row['label']), dtype=torch.long)  # 单标签分类用 long 类型
        cfp_image = Image.open(row['cfp']).convert('RGB')
        oct_image = Image.open(row['oct']).convert('RGB')
        cfp = self.transform(cfp_image)
        oct = self.transform(oct_image)
        return cfp, oct, label



if __name__ == '__main__':
    csv_path=r"/dataSet/OLIVES/trainValTest/val.xlsx"
    args = SimpleNamespace(**{
        "input_size": 224,
        "data_path" : "/dataSet/OLIVES/images/",
        "images_number":1 ,
        "in_domains": "cfp"
    })
    dataset_val=EyeDiseaseDataset(args,csv_path, transform=build_transform(is_train='val',args= args))

    data_loader = DataLoader(
        dataset_val,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=False
    )
    for batch in data_loader:
        cfp = batch[0]
        # oct = batch[1]
        labels = batch[-1]
        print("Batch cfp shape:", cfp.shape)
        # print("Batch oct shape:", oct.shape)
        print("Batch labels shape:", labels.shape)


