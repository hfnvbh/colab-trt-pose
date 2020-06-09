import sys
sys.path.append('trt_pose')
sys.path.append('torch2trt/torchtrt')

import json
import torch

import torch2trt
from torch2trt import TRTModule

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

import cv2
import torchvision.transforms as transforms
import PIL.Image

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

cam = cv2.VideoCapture('youtube.mp4')
output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'),
                         int(cam.get(cv2.CAP_PROP_FPS)), 
                         (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))))

def execute(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    return image


ret,img = cam.read()
while(ret):
    imag = execute(img)
    output.write(imag)
    ret,img = cam.read()