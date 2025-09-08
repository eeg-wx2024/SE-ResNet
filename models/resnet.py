from torchvision.models import ResNet
import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMLayer(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CALayer(nn.Module):
    def __init__(self, inplace, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden_dim = max(8, inplace // reduction)
        self.conv1 = nn.Conv2d(inplace, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(hidden_dim, inplace, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, inplace, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_h * a_w
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def get_act_type(act_name):
    if act_name == "elu":
        return nn.ELU(inplace=True)
    elif act_name == "silu":
        return nn.SiLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def get_attn_type(
    attn_name,
    planes,
):
    if attn_name == "se":
        reduction = 16
        return SELayer(planes, reduction)
    elif attn_name == "cbam":
        return CBAMLayer(planes)
    elif attn_name == "ca":
        return CALayer(planes)
    else:
        return nn.Identity()


def get_basic_block(attn_name, act_name):

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
        ):
            super().__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)

            self.bn1 = nn.BatchNorm2d(planes)

            self.act = get_act_type(act_name)

            self.conv2 = conv3x3(planes, planes, 1)
            self.bn2 = nn.BatchNorm2d(planes)

            self.attn = get_attn_type(attn_name, planes)

            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.attn(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.act(out)

            return out

    return BasicBlock


def get_bottleneck_block(attn_name, act_name):
    class BottleneckBlock(nn.Module):
        expansion = 4

        def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.act = get_act_type(act_name)
            self.attn = get_attn_type(attn_name, planes * 4)

            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.act(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.attn(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.act(out)

            return out

    return BottleneckBlock


def resnet18(attn_name=None, act_name=None):
    block_type = get_basic_block(attn_name, act_name)
    model = ResNet(block_type, [2, 2, 2, 2], num_classes=40)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.relu = nn.ELU(inplace=True)
    return model


def resnet(attn_name=None, act_name=None, blocks=[2, 2, 2, 2]):
    block_type = get_basic_block(attn_name, act_name)
    model = ResNet(block_type, blocks, num_classes=40)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def resnet50(attn_name=None, act_name=None):
    block_type = get_bottleneck_block(attn_name, act_name)
    model = ResNet(block_type, [2, 2, 2, 2])
    return model


def resnet10(attn_name=None, act_name=None):
    block_type = get_basic_block(attn_name, act_name)
    model = ResNet(block_type, [1, 1, 1, 1])
    return model


if __name__ == "__main__":

    model = resnet(attn_name="se", act_name="silu", blocks=[1, 2, 2, 1])
    print(model)
