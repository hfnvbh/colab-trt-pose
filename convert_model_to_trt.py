import sys
sys.path.append('trt_pose')
sys.path.append('torch2trt/torchtrt')

import json
import torch

import torch2trt

import trt_pose.coco
import trt_pose.models


with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS))

WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)