import os
import torch
import sys
import time
import pandas as pd
import numpy as np
import cv2
import torch.nn as nn
from PIL import Image
from model import vgg
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self,  img_dir, label_dir, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir # ./data/train/vis
        self.label_dir = label_dir # ./data/train/infra
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_list = os.listdir(self.img_dir)
        img_path = os.path.join(self.img_dir, img_list[idx])
        image = read_image(img_path) / 255.0

        label_path = os.path.join(self.label_dir, img_list[idx])
        label = read_image(label_path) / 255.0
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

def customdataset(img_dir, label_dir, **kwargs):
    dataset = CustomImageDataset(img_dir, label_dir, **kwargs)
    return dataset

def ShowImg(dataset):
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    img, label = dataset[sample_idx]
    img = img.numpy()  # 将tensor数据转为numpy数据
    label = label.numpy()

    maxValue = img.max()
    img = img * 255 / maxValue  # normalize，将图像数据扩展到[0,255]
    maxValue2 = label.max()
    label = label * 225 / maxValue2
    mat = np.uint8(img)  # float32-->uint8
    mat2 = np.uint8(label)

    mat = mat.transpose(1, 2, 0)  # mat_shape: (224, 224，3)
    mat2 = mat2.transpose(1, 2, 0)
    cv2.imshow("img", mat)
    cv2.imshow("label", mat2)
    cv2.waitKey()
    cv2.imwrite(f"./img{sample_idx}.jpg", mat)
    cv2.imwrite(f"./label{sample_idx}.jpg", mat2)

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((256)),
                                     transforms.CenterCrop((168, 224))]),
        "val": transforms.Compose([transforms.Resize((256)),
                                     transforms.CenterCrop((168, 224))])
    }

    train_dir = "../llvip/train/"
    val_dir = "../llvip/test/"

    # 构建train数据集
    train_dataset = customdataset(img_dir = os.path.join(train_dir, "vis"),
                                  label_dir = os.path.join(train_dir, "infra"),
                                  transform = data_transform["train"])
    train_num = len(train_dataset)
    # 构建validate数据集
    validate_dataset = customdataset(img_dir = os.path.join(val_dir, "vis"),
                                     label_dir = os.path.join(val_dir, "infra"),
                                     transform = data_transform["val"])
    val_num = len(validate_dataset)

    batch_size = 2
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    # 加载train数据
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    # 加载train数据
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))



    model_name = "vgg16"
    net = vgg(model_name = model_name)
    weight_path = '../vgg16-pre.pth'

    # 抽出预训练模型中的K,V
    pretrain_model = torch.load(weight_path, map_location=device)
    # 抽出现有模型中的K,V
    model_dict = net.state_dict()
    # 新建权重字典，并更新
    state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
    # 更新现有模型的权重字典
    model_dict.update(state_dict)
    # 载入更新后的权重字典
    net.load_state_dict(model_dict)
    # 冻结权重，即设置该训练参数为不可训练即可
    for name, para in net.named_parameters():
        if name in state_dict:
            para.requires_grad = False

    net.to(device)
    # print(net)
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir = "run/normal_experience")
    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(net, init_img)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    best_acc = 1.0
    checkpoint_interval = 5
    save_path = './weight/{}Net.pth'.format(model_name)
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()  # 启用dropout
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # 每五个epoch设一个checkpoint，保存至checkpoints文件
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "./checkpoints/checkpoint_{}_epoch.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        # validate
        net.eval()  # 关闭dropout
        acc = 0.0  # accumulate accurate number / epoch
        abs_sum = torch.tensor([0.0])
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                dif = outputs - val_labels.to(device)
                abs_td = torch.sum(torch.abs(dif), (1,2,3)) / (dif[0,:,:,:].numel())
                abs_sum = torch.cat([abs_sum.to(device),abs_td], dim=0)

                # predict_y = torch.max(outputs, dim=1)[1]
                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_abs_std = torch.sum(abs_sum) / val_num
        # val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_abs_std))

        # 在tensorboard中写入loss， abs diff， learning rate的曲线图
        tags = ["train_loss", "absolute difference", "learning rate"]
        tb_writer.add_scalar(tags[0], running_loss/train_steps, epoch)
        tb_writer.add_scalar(tags[1], val_abs_std, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # 在log文件中保存日志：epoch | MSE | abs
        log = f'\nepoch: {epoch} | MSE:{running_loss / train_steps}| abs:{val_abs_std}'
        with open(f'./log/{model_name}n_logs.txt', 'a') as f:
            f.write(log + '\n')

        # 在weight文件中，保存最好的权重文件
        if val_abs_std < best_acc:
            best_acc = val_abs_std
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ == '__main__':
    main()

