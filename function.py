import torch
import numpy as np
from PIL import Image
import os
import numpy as np
from torch.nn import functional as F
import random
import scipy.stats as st

def gkern(kernlen, nsig):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
class GaussianBlur(torch.nn.Module):
    def __init__(self,kernel):
        super(GaussianBlur, self).__init__()
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False).cuda()
        self.padding = int(kernel.shape[-1]-1)/2
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight,padding=int(self.padding))
        x2 = F.conv2d(x2.unsqueeze(1), self.weight,padding=int(self.padding))
        x3 = F.conv2d(x3.unsqueeze(1), self.weight,padding=int(self.padding))
        x = torch.cat([x1, x2, x3], dim=1)
        return x


def obtain_pair(pair_path):
    """
    input:The path of the txt file's path
    output:The dict which key is the attack picture's name,
           and the value is the corresponding attacked picture's name.
    """
    pair_dict = {}
    with open(pair_path,'r') as file:
        for line in file:
            line = line.rstrip()
            pair = line.split(" ")
            pair_dict[pair[0]]=pair[1]
    return pair_dict
def input_diversity(image, low=102, high=112):
    """
    input:The image need to be processed,type is torch.tensor(),size is (3,112,112)
    parameters:
        low:The minimun size of the images
        high:The maximum size of the images
    output:The image been processed by SIM. The picture first be scaled in the scale.
           Then the images is been padded to the size of (112,112)
    """
    if random.random() > 0.9:
        return image
    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded
class Ensemble(torch.nn.Module):
    """
    input:The list of the all models
    output:The list of the features getted by the all models
    """
    def __init__(self,model_all):
        super(Ensemble, self).__init__()
        self.model_all = model_all

    def forward(self, x):
        logits = []
        for model in self.model_all:
            q = model(x)
            q = tensorNorm(q)
            logits.append(q)
        return logits
class HParams:
    """
    Get the configs of the IRSE101 model.
    """
    def __init__(self):
        self.pretrained = False
        self.use_se = True

def img2tensor(img):
    """
    input:  The image data,the type is np.array(),and the size is [H,W,C]
    output: The normalized image data,the type is torch.tensor().And the size is [B,C,H,W]
            The value is normalized into the scale (-1,1)
    """
    img = np.array(img,dtype=np.float32)
    img = np.transpose(img,[2,0,1])
    img = (img-127.5)/128
    img = torch.from_numpy(img).unsqueeze(0)
    return img

def tensorNorm(ts):
    """
    input:The feature vectors getted by the model
    output:The normalized feature vectors,whose 2norm is 1.
    """
    norm = torch.norm(ts,dim=1)
    res = torch.div(ts,norm)
    return res






