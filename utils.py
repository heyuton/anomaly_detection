import os
import albumentations  #这个包就是图像裁剪包    https://blog.csdn.net/u014264373/article/details/114144303
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)   #图像裁剪
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)              #这里的size会影响到输出图像的尺寸
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader

#***********************这里为自定义的装载函数，用于装载aim_images*******************************
def load_another_aim_data(args):
    train_data = ImagePaths(args.another_dataset_aim_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader

def load_aimdata(args):
    aim_data = ImagePaths(args.aim_dataset_path, size=256)
    aim_loader = DataLoader(aim_data, batch_size=args.batch_size, shuffle=False)
    return aim_loader

#我这里需要想模型里输入图像三元组，所以需要装载三个不同数据集的图像：I，T，R

def confirm_generatorname(args):
    generator_name = args.generatorname
    return generator_name

#******************************************************************************************



# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.           下面的模块是编码器和解码器的函数，与裁剪图像无关
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    #plt.show()
