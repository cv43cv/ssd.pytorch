import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from layers import *
from data import voc, coco
import os

def build_ipssd(phase, size=300, num_classes=21, pretrained=True):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    base = resnet(pretrained)
    extras = resadd_extras(1024, size)
    loc, conf = boxbox(base, extras, num_classes)
    model = ipSSD(
            phase=phase,
            size=size,
            base = base,
            extras = extras,
            loc = loc,
            conf = conf,
            num_classes = num_classes,
            pretrained = pretrained
        )
    return model



class ipSSD(nn.Module):

    def __init__(self, phase, size, base, loc, conf, extras, num_classes, pretrained=True):
        super(ipSSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size


        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        #
        #for i in range(4):
        #    for p in self.base[i].parameters():
        #        p.requires_grad = False
        #

    def forward(self, x):
        """
        Args:
            x: input image or batch of images shape (batch,3,300,300)
        """
        sources = []
        loc = []
        conf = []

        for k in range(6):
            x = self.base[k](x)

        sources.append(x)
        
        for k in range(6, 8):
            x = self.base[k](x)

        sources.append(x)
        
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if self.phase == 'test':
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                nn.Softmax(dim=-1)(conf.view(conf.size(0), -1,
                             self.num_classes)),               # conf preds
                self.priors.type(type(x.data))
            )
        
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(loc.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class AtrousBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AtrousBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=2, stride=stride,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet(pretrained = True):

    resnet50 = models.resnet50(pretrained=pretrained)
    layers = []
    layers += [resnet50.conv1]
    layers += [resnet50.bn1]
    layers += [resnet50.relu]
    layers += [resnet50.maxpool]
    layers += [resnet50.layer1]
    layers += [resnet50.layer2]
    layers += [resnet50.layer3]
    resnet50.inplanes = 1024
    layer4 = resnet50._make_layer(AtrousBottleneck, 256, 3)
    for m in layer4:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    layers += [layer4]

    return layers


def resadd_extras(i, size):
    # Extra layers added to VGG for feature scaling
    if size == 300:
        cfg = [256, 'S', 512, 256, 'S', 512, 128, 256, 128, 256]
    elif size == 512:
        cfg = [256, 'S', 512, 256, 'S', 512, 128, 'S', 256, 128, 'S', 256]
 
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def boxbox(inception, extra_layer, num_classes):
    cfg =[4, 6, 6, 6, 4, 4] # number of boxes per feature map location

    loc_layers = []
    conf_layers = []
    in_c = [512, 1024]

    for k, v in enumerate(in_c):
        loc_layers += [nn.Conv2d(v, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v, cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layer[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return (loc_layers, conf_layers)