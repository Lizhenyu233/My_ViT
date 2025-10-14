import os
import json

import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from ViT import VisionTransformer

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 测试图像处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),    # 调整图像大小
         transforms.CenterCrop(224),# 中心裁剪到224x224（ViT标准输入尺寸）
         transforms.ToTensor(),     # 转换为PyTorch张量，数值范围[0,1]
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])# 归一化到[-1,1]范围

    # load image
    # img_path = args.img_path
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # # 显示原始图像
    # # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # 在第0维添加batch_size维度，变为[1, C, H, W]
    # img = torch.unsqueeze(img, dim=0)

    # read class_indices
    # 加载训练时生成的类别索引文件
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indices = json.load(f)
    
    # 加载图像
    # 遍历文件夹，一个文件夹对应一个类别
    root = args.testdata_path
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    test_images_path = [] # 测试图像路径
    test_images_label = []# 测试图像类别
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 分配到测试集
        test_images_path.append(images)
        test_images_label.append(image_class)

    # create model
    model = VisionTransformer(args.img_size, 
                              args.patch_size,
                              args.in_c,
                              args.num_classes,
                              args.embed_dim,
                              args.depth,
                              args.num_heads,
                              args.mlp_ratio,
                              args.qkv_bias,
                              args.qk_scale,
                              args.representation_size,
                              args.drop_ratio,
                              args.attn_drop_ratio).to(device)
    # load model weights
    model_weight_path = args.model_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    model.eval()
    # 保存测试结果
    test_result = []
    with torch.no_grad():   # 禁用梯度计算，节省内存和计算资源
        # predict class
        for i in range(len(test_images_path)):
            test_result.append([])
            for img_path in test_images_path[i]:
                img = Image.open(img_path)
                img = data_transform(img)
                # 在第0维添加batch_size维度，变为[1, C, H, W]
                img = torch.unsqueeze(img, dim=0)
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)  # 将输出转换为概率分布
                predict_cla = torch.argmax(predict).numpy() # 获取最大概率对应的类别索引
                test_result[i].append(predict_cla) # 保存测试结果
    # 测试结果可视化
    # 保存正确的测试结果，以及错误的测试图片
    test_label_true = []
    error_images_path = []
    error_images_label = []
    label_to_class = dict((val, key) for key, val in class_indices.items())
    for i in range(len(test_result)):
        test_label_true.append([])
        for j, label in enumerate(test_result[i]):
            if label == test_images_label[i]:
                test_label_true[i].append(label)
            else:
                error_images_path.append(test_images_path[i][j])
                error_images_label.append(label_to_class[int(label)])

    # 统计每个类别预测正确的数量
    true_every_class_num = [len(x) for x in test_label_true]
    # 绘制测试结果树状图
    plt.bar(range(len(test_result)), true_every_class_num, align='center')
    # 将横坐标0,1,2,3,4替换为相应的类别名称
    plt.xticks(range(len(flower_class)), flower_class)
    # 在柱状图上添加数值标签
    for i, v in enumerate(true_every_class_num):
        plt.text(x=i, y=v + 0.1, s=str(v), ha='center')
    # 设置x坐标
    plt.xlabel('image class')
    # 设置y坐标
    plt.ylabel('number of true result')
    # 设置柱状图的标题
    plt.title('test result distribution')
    plt.savefig("test_result.png")
    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.savefig("test_result.png")
    #plt.show()
    # 测试错误的图片
    print("测试错误的图片路径：{}".format(error_images_path))
    print("测试错误的图片类型：{}".format(error_images_label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 网络参数
    parser.add_argument('--img_size', type=int, default=224, help="输入图像大小")
    parser.add_argument('--patch_size', type=int, default=16, help="分割的patch大小")
    parser.add_argument('--in_c', type=int, default=3, help="输入图像通道数")
    parser.add_argument('--num_classes', type=int, default=5, help="类别数量")
    parser.add_argument('--embed_dim', type=int, default=768, help="编码维度，16x16x3=768")
    parser.add_argument('--depth', type=int, default=12, help="tf的encoder堆叠层数")
    parser.add_argument('--num_heads', type=int, default=12, help="tf的注意力头数量")
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help="MLP结构中的膨胀系数")
    parser.add_argument('--qkv_bias', type=bool, default=True, help="启用qkv偏置功能")
    parser.add_argument('--qk_scale', type=int, default=None, help="计算qk分数时的分母缩放系数")
    parser.add_argument('--representation_size', type=int, default=None, help="重新表征大小")
    parser.add_argument('--drop_ratio', type=float, default=0., help="除attention层外其余层的丢弃概率")
    parser.add_argument('--attn_drop_ratio', type=float, default=0., help="attention层中的丢弃概率")

    # parser.add_argument('--img_path', type=str, default="./testdata/6953297_8576bf4ea3.jpg", help="测试图片路径")
    parser.add_argument('--model_path', type=str, default="train/final_model.pth")
    parser.add_argument('--testdata_path', type=str, default='./testdata')
    args = parser.parse_args()
    test()