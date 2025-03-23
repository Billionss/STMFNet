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
    def __init__(self, block, num_blocks, in_channels):
        super(ResNet, self).__init__()
        self.channels = 8
        self.in_channels = in_channels
        # self.strat_conv = nn.Conv2d(1, 3, kernel_size=1)  # 修改卷积核大小和步长以适应224x224的
        self.conv1 = nn.Conv2d(in_channels, self.channels, kernel_size=7, stride=2, padding=3, bias=False)  # 修改卷积核大小和步长以适应224x224的输入
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.channels, out_channels, stride))
            self.channels = out_channels
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

def ResNet18( in_channels):
    return ResNet(BasicBlock, [1,1,1,1], in_channels)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

# torch.Size([3, 64, 112, 112])
# torch.Size([3, 128, 56, 56])
# torch.Size([3, 256, 28, 28])
# torch.Size([3, 512, 14, 14])


class MultiScaleConc(nn.Module):

    def __init__(self, in_channels, channels):
        super(MultiScaleConc, self).__init__()
        # define channels before using it
        self.start_conv = nn.Conv2d(in_channels, channels//8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(channels//8)

        self.scale_conv1 = nn.Conv2d(channels//8 , channels//8, kernel_size=1, stride=1) # 4
        self.scale_conv2 = nn.Conv2d(channels//8 , channels//4, kernel_size=2, stride=2) # 8
        self.scale_conv3 = nn.Conv2d(channels//8 , channels//2, kernel_size=4, stride=4) # 16
        self.scale_conv4 = nn.Conv2d(channels//8 , channels//1, kernel_size=8, stride=8) # 32

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
        x = torch.squeeze(self.padding(x))

        x = self.bn(self.start_conv(x))

        c1 = self.scale_conv1(x)
        c2 = self.scale_conv2(x)
        c3 = self.scale_conv3(x)
        c4 = self.scale_conv4(x)

        return [c1, c2, c3, c4]


class BiStreamFPN(nn.Module):
    def __init__(self, in_channels, channels):
        super(BiStreamFPN, self).__init__()

        self.conv_net = MultiScaleConc(in_channels, channels)
        
        self.top_down1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.top_down2 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        self.top_down3 = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=0)
        self.top_down4 = nn.Conv2d(4, 32, kernel_size=1, stride=1, padding=0)

        # self.down_top1 = nn.Conv2d(4, 32, kernel_size=1, stride=1, padding=0)
        # self.down_top2 = nn.Conv2d(8, 32, kernel_size=1, stride=1, padding=0)
        # self.down_top3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        # self.down_top4 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):
        c2, c3, c4, c5 = self.conv_net(x)  # channels :[4 8 16 32]
        # c2 [4, 112, 112]
        # c3 [8, 56, 56]
        # c4 [16, 28, 28]
        # c5 [32, 14, 14]

        # top down stream
        p5 = self.top_down1(c5)
        p4 = self.upsample(p5) + self.top_down2(c4)
        p3 = self.upsample(p4) + self.top_down3(c3)
        p2 = self.upsample(p3) + self.top_down4(c2)
        
        # t2 = self.down_top1(c2)
        # t3 = F.interpolate(t2, scale_factor=0.5, mode='nearest') + self.down_top2(c3)
        # t4 = F.interpolate(t3, scale_factor=0.5, mode='nearest') + self.down_top3(c4)
        # t5 = F.interpolate(t4, scale_factor=0.5, mode='nearest') + self.down_top4(c5)

        # return [ t2, t3, t4, t5]
        return [p2, p3, p4, p5]

class MultiScaleFusion(nn.Module):
    def __init__(self):
        super(MultiScaleFusion, self).__init__()
        self._1conv = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self._2conv = nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0)
        self._4conv = nn.Conv2d(32, 32, kernel_size=4, stride=4, padding=0)
        self._8conv = nn.Conv2d(32, 32, kernel_size=8, stride=8, padding=0)

        self.mlp = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1, x2, x3, x4 = x
        x1 = self._8conv(x1)
        x2 = self._4conv(x2)
        x3 = self._2conv(x3)
        x4 = self._1conv(x4)
        
        out = self.mlp(torch.cat((x1, x2, x3, x4), dim=1))
        
        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dims):
        super(Conv, self).__init__()

        self.start_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(14,14), stride=(7, 7))
        self.linear = nn.Linear(31*31, dims)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

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
        x = self.padding(x).squeeze(2)
        # x = x.reshape(x.shape[0],  -1, x.shape[2], x.shape[3])
        x = self.bn(self.start_conv(x))
        x = self.relu(x)
        x = self.linear(x.reshape(x.shape[0], x.shape[1], -1))

        return x


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
    
    fusion = MultiScaleFusion()
    
    out = fusion(output)
    print(type(out))
    print(out.shape)
    
