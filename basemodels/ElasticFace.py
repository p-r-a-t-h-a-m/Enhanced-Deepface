# built-in dependencies
import os
import torch
from torch import nn

# 3rd party dependencies
import gdown

# project dependencies
from deepface.commons import folder_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons import logger as log


class SEModule(nn.Module):
    """Implementation of Squeeze-and-Excitation Module."""
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x

class IBasicBlock(nn.Module):
    """Basic building block for the IResNet architecture."""

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1,use_se=False):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride
        self.use_se=use_se
        if (use_se):
         self.se_block=SEModule(planes,16)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if(self.use_se):
            out=self.se_block(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """Main IResNet architecture."""
    """The IResnet model is heavily inspired from github.com/fdbtrs/ElasticFace/tree/main/backbones 
    """
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, use_se=False):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.use_se=use_se
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # Define initial layers
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        # Define layers for each stage
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2 ,use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1] ,use_se=self.use_se)
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2] ,use_se=self.use_se)
        # Final layers
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout =nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,use_se=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation,use_se=use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("Input size:", x.size())  # Initial input size
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        # print("After layer1 size:", x.size())  # Output size after layer1
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print("After layer4 size:", x.size())  # Output size after the last conv layer
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        # print("After flatten size:", x.size())  # Size before fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

def iresnet100(**kwargs):
    """Constructs an IResNet-100 model."""
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)

logger = log.get_singletonish_logger()

# Pre-trained weights URLs
PRETRAINED_WEIGHTS = {
    "ElasticFace_Arc": "https://drive.google.com/uc?id=1dYX9muI63Su4aBFzdYXSDQXp4FriijC2",   #For Arc

    "ElasticFace_Arcplus": "https://drive.google.com/uc?id=1rQ7H8cXxpTf21qpIBJP7ZiibfntPV4TY",  #For Arc+

    "ElasticFace_Cos": "https://drive.google.com/uc?id=1eUwnbXR5pSvq0e63DYSo71nAR6-p-mzc", #For Cos

    "ElasticFace_Cosplus": "https://drive.google.com/uc?id=1nTMBi2tVUOjHxbm2m9NMKGVpfLNk9Vil" #For Cos+
}

# Output weight file names
OUTPUT_FILES = {
    "ElasticFace_Arc": "elasticface_arc_v1.pth",  #For Arc

    "ElasticFace_Arcplus": "elasticace_Arc+_v1.pth",  #For Arc+

    "ElasticFace_Cos": "elasticface_cos_v1.pth",   #For Cos

    "ElasticFace_Cosplus": "elasticace_Cos+_v1.pth"  #For Cos+
}

class ElasticFaceModel(FacialRecognition):
    """Base class for ElasticFace model."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.input_shape = (112, 112)
        self.output_shape = 512
        self.model = self.load_model()
    
    def load_model(self):
        """Load the IResNet-100 model with pretrained weights."""
        model = iresnet100(num_features=512)

        # Determine output path for pretrained weights
        home = folder_utils.get_deepface_home()
        output = os.path.join(home, ".deepface", "weights", OUTPUT_FILES[self.model_name])

        # Download pretrained weights if they don't exist locally
        if not os.path.isfile(output):
            logger.info(f"Pre-trained weights are being downloaded from {PRETRAINED_WEIGHTS[self.model_name]} to {output}")
            gdown.download(PRETRAINED_WEIGHTS[self.model_name], output, quiet=False)
            logger.info(f"Pre-trained weights have been downloaded to {output}")

        # Loading pretrained weights into the model
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        state_dict = torch.load(output, map_location=map_location)

        print("Keys and sizes in state_dict:")
        for key, value in state_dict.items():
            print(f"{key}: {value.size()}")

        model.load_state_dict(state_dict)
        model.eval()
        # model.to(device='cuda') # if using gpu

        return model

class ElasticFace_ArcClient(ElasticFaceModel):
    """Client class for ElasticFace_Arc model."""
    
    def __init__(self):
        super().__init__("ElasticFace_Arc")

class ElasticFace_ArcplusClient(ElasticFaceModel):
    """Client class for ElasticFace_Arc+ model."""
    
    def __init__(self):
        super().__init__("ElasticFace_Arcplus")

class ElasticFace_CosClient(ElasticFaceModel):
    """Client class for ElasticFace_Cos model."""
    
    def __init__(self):
        super().__init__("ElasticFace_Cos")

class ElasticFace_CosplusClient(ElasticFaceModel):
    """Client class for ElasticFace_Cos+ model."""
    
    def __init__(self):
        super().__init__("ElasticFace_Cosplus")