from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    # 必须实现，PyTorch DataLoader通过这个方法知道数据集的大小
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # 使用PIL库打开图像文件
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod   # 不需要访问类实例的方法
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # zip(*batch): 将[(img1, label1), (img2, label2), ...]解包为([img1, img2, ...], [label1, label2, ...])
        images, labels = tuple(zip(*batch))
        '''torch.stack: 将多个张量沿着新维度堆叠
            dim=0: 在第0维（批次维度）堆叠
            将[C,H,W]形状的单个图像张量堆叠为[B,C,H,W]的批次张量'''
        images = torch.stack(images, dim=0)
        # 将Python列表转换为PyTorch张量
        labels = torch.as_tensor(labels)
        return images, labels