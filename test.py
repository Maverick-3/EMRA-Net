import os
import argparse
import torch
import torchvision
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from model import DehazeNet


parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument('--nChannel', type=int, default=3)
parser.add_argument('--nFeat', type=int, default=64)

parser.add_argument("--ckp_dir", type=str, default='./model/pretrained.pth')

parser.add_argument("--test_dir", default='./test', type=str)
parser.add_argument('--output_dir', type=str, default='./results')
opt = parser.parse_args()

if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

# --------------------------------------------------------------------------
net = DehazeNet(opt).cuda()
net.load_state_dict(torch.load(opt.ckp_dir))
net.eval()


images = os.listdir(opt.test_dir)

if __name__ == "__main__":
    for idx in images:
        img = Image.open(os.path.join(opt.test_dir, idx))
        h, w = img.size
        img = ToTensor()(img)
        img = Variable(img)
        img = img.view(1, -1, w, h)
        img = img.cuda()
        with torch.no_grad():
            img = net(img)
        torchvision.utils.save_image(img, os.path.join(opt.output_dir, idx))

