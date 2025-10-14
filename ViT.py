from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    '''Patch Embedding 层'''
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_c=3, 
                 embed_dim=768, 
                 norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # 确保图像尺寸为元组形式
        patch_size = (patch_size, patch_size)  # patch尺寸
        self.img_size = img_size
        self.patch_size = patch_size
        
        # 计算网格大小和patch数量
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 14x14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 196个patch
        
        # 投影层：将图像patch转换为嵌入向量
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 归一化层，如果未提供则使用恒等映射
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 三维变换 (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 196, 768)
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)  
        x = self.norm(x)  # 归一化
        return x
    
def _init_vit_weights(m):
    '''权重初始化函数'''
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)  # 截断正态分布初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 偏置初始化为0
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")  # He初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)  # LayerNorm偏置初始化为0
        nn.init.ones_(m.weight)  # LayerNorm权重初始化为1

class Attention(nn.Module):
    '''Multi-Head Attention 机制'''
    def __init__(self, 
                 dim, # token的dim
                 num_heads=8, 
                 qkv_bias=False, #生成qkv时是否使用偏置
                 qk_scale=None, 
                 attn_drop_ratio=0., 
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子
        
        # QKV投影层：同时计算query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #全连接层Linear
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)  # 计算后输出投影
        self.proj_drop = nn.Dropout(proj_drop_ratio)  # 输出dropout

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, num_patches+1, embed_dim]
    
        # 生成QKV
        # (B, N, C) -> (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离Q,K,V
    
        # 计算注意力分数
        # transpose:-> [B, num heads, head_dim, N]
        # @:multiply -> [B, num heads, N, N]
        # 若是多维，矩阵乘法只对最后两维度生效
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Q·K^T / sqrt(d_k)
        attn = attn.softmax(dim=-1)  # 对于每行进行Softmax归一化
        attn = self.attn_drop(attn)  # 应用dropout
    
        # 加权求和
        # @:multiply -> [B, num heads, N, head_dim]
        # transpose:-> [B, N, num heads, head_dim]
        # reshape:-> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 合并多头输出
        x = self.proj(x)  # 线性投影
        x = self.proj_drop(x)  # 输出dropout
        return x
    
class MLP(nn.Module):
    '''MLP层'''
    def __init__(self, 
                 in_features, # 输入维度
                 hidden_features=None, #一般默认是输入维度的4倍
                 out_features=None, 
                 act_layer=nn.GELU, # 激活函数
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 默认输出维度等于输入
        hidden_features = hidden_features or in_features  # 隐藏层维度
        
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层
        self.act = act_layer()  # 激活函数（默认GELU）
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    '''Transformer的Emcoder Block'''
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., # 节点个数是输入节点个数的4倍
                 qkv_bias=False, 
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0., 
                 drop_out_ratio=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # 第一个LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, 
                             proj_drop_ratio=drop_ratio)  # 注意力机制
        self.drop_out = nn.Dropout(p=drop_out_ratio)  # 残差连接的dropout
        self.norm2 = norm_layer(dim)  # 第二个LayerNorm
        
        # MLP模块：隐藏层维度 = 输入维度 × mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop_ratio)
    def forward(self, x):
        # 第一个残差块：LayerNorm → Attention → Dropout → 残差连接
        x = x + self.drop_out(self.attn(self.norm1(x)))
        # 第二个残差块：LayerNorm → MLP → Dropout → 残差连接  
        x = x + self.drop_out(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    '''ViT整体'''
    def __init__(self, img_size=224, patch_size=16, in_c=3, 
                 num_classes=6, # 有多少类
                 embed_dim=768, 
                 depth=12, #Block堆叠的个数
                 num_heads=12, 
                 mlp_ratio=4.0, 
                 qkv_bias=True, 
                 qk_scale=None, 
                 representation_size=None, # Pre-logits层的节点个数，ViT中不需要搭建
                 drop_ratio=0, 
                 attn_drop_ratio=0, 
                 embed_layer=PatchEmbed):
        super(VisionTransformer, self).__init__()
        
        # 基础参数设置
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_tokens = 1  # CLS token
        
        # 层定义
        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # 创建LayerNorm偏函数
        act_layer = nn.GELU  # 激活函数
        
        # Patch Embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, 
                                      in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS Token和位置编码（可学习参数）
        # 第一个纬度的1是batch_size纬度
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        
        # Transformer Blocks堆叠
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)  # 堆叠depth个Block
        ])
        
        # 输出层
        self.norm = norm_layer(embed_dim)
        
        # Pre-logits层（可选）
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())  # Tanh激活
            ]))
        else: # 默认不做处理
            self.has_logits = False
            self.pre_logits = nn.Identity()  # 恒等映射
        
        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)  # 应用自定义初始化

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x)  # [B, 196, 768]
    
        # 2. 添加CLS Token
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 扩展CLS token
        # 第二个维度上拼接
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
    
        # 3. 添加位置编码并应用dropout
        x = self.pos_drop(x + self.pos_embed)
    
        # 4. 通过Transformer Blocks
        x = self.blocks(x)
    
        # 5. 最终LayerNorm
        x = self.norm(x)
    
        # 6. 使用CLS token进行分类
        x = self.pre_logits(x[:, 0])  # 取CLS token对应的输出
    
        # 7. 分类头
        x = self.head(x)
        return x
    
# 测试
if __name__ == '__main__':
    model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768,
                              depth=12, num_heads=12, representation_size=None,
                              num_classes=5)
    print(model)