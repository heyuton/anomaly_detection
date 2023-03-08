import torch
import os
from tmp import CustomImageDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

def customdataset(img_dir, label_dir, **kwargs):
    dataset = CustomImageDataset(img_dir, label_dir, **kwargs)
    return dataset


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        X = X.float()
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

train_dir = "../LLVIP/train/"
# train_dataset = ImageFolder(root=train_dir, transform=None)
train_dataset = customdataset(img_dir = os.path.join(train_dir, "vis"),
                              label_dir = os.path.join(train_dir, "infra"),)
val_dir = "../LLVIP/test/"
# val_dataset = ImageFolder(root=val_dir, transform=None)
val_dataset = customdataset(img_dir = os.path.join(val_dir, "vis"),
                              label_dir = os.path.join(val_dir, "infra"))

print(getStat(train_dataset), "\n")
print(getStat(val_dataset))