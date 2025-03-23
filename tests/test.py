import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # self.strat_conv = nn.Conv2d(1, 3, kernel_size=1)  # 修改卷积核大小和步长以适应224x224的
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改卷积核大小和步长以适应224x224的输入
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def padding(self, x):
        # 计算需要填充的像素数
        padding_height = 224 - x.size(3)
        padding_width = 224 - x.size(4)

        # 计算在每个方向上需要填充的像素数
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        # 使用pad函数填充图像
        x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode='constant', value=0)
        return x
    
    def forward(self, x):
        x = torch.squeeze(self.padding(x))   # 使用padding函数填充输入图像
        # x = F.interpolate(x, size=(224, 224), mode='constant', align_corners=False)  # 将输入图像大小使用0值均匀扩充至(224, 224)大小
        
        x = F.relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        return [feat1, feat2, feat3, feat4]  # 返回各层不同尺度的特征值组成的列表

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

# torch.Size([3, 64, 112, 112])
# torch.Size([3, 128, 56, 56])
# torch.Size([3, 256, 28, 28])
# torch.Size([3, 512, 14, 14])


class BiStreamFPN(nn.Module):
    def __init__(self):
        super(BiStreamFPN, self).__init__()
        self.resnet = ResNet18()
        
        self.top_down1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.top_down2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.top_down3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.top_down4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        
        self.down_top1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.down_top2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.down_top3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.down_top4 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        c2, c3, c4, c5 = self.resnet(x)  # channels : 64, 128, 256, 512
        # top down stream
        p5 = self.top_down1(c5) 
        p4 = self.upsample(p5) + self.top_down2(c4)
        p3 = self.upsample(p4) + self.top_down3(c3)
        p2 = self.upsample(p3) + self.top_down4(c2)
        
        t2 = self.down_top1(c2)
        t3 = F.interpolate(t2, scale_factor=0.5, mode='nearest') + self.down_top2(c3)
        t4 = F.interpolate(t3, scale_factor=0.5, mode='nearest') + self.down_top3(c4)
        t5 = F.interpolate(t4, scale_factor=0.5, mode='nearest') + self.down_top4(c5)
        
        return [ p2 + t2, p3 + t3, p4 + t4, p5 + t5]

if __name__ == '__main__':
    
    # 实例化ResNet18模型
    model = BiStreamFPN()

    # 打印模型结构
    # print(model)

    # 创建一个随机的输入张量，模拟一批次的图像数据
    input = torch.randn(64, 3, 1, 162, 209)

    output = model(input)

    # 打印输出数据
    print(type(output))
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
