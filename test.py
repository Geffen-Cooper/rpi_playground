import time

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image
import pandas as pd


df = pd.read_csv("mapping.txt",sep=":",header=None)
df[1] = df[1].apply(lambda row: row.split(',')[0].replace("'",""))
class_name_map = list(dict(zip(df[0],df[1])).values())

#print(torch.__version__)
torch.backends.quantized.engine = 'qnnpack'

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

t_net = models.efficientnet_b0(weights='DEFAULT')
s_net = models.shufflenet_v2_x0_5(weights='DEFAULT')
# net = models.shufflenet_v2_x0_5()
# net = models.mobilenet_v2(num_classes=10,width_mult=0.1)
#net = models.mobilenet_v3_large(width_mult=0.5)
# jit model to take it from ~20fps to ~30fps
s_net = torch.jit.script(s_net)
t_net = torch.jit.script(t_net)

started = time.time()
last_logged = time.time()
frame_count = 0
torch.set_num_threads(1)
m = "s"
p = 1.0
rands = torch.rand(110)
fps = 0
with torch.no_grad():
    start = time.time()
    i = 0
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")
        image = cv2.rotate(image,cv2.ROTATE_180)
        #cv2.imshow('img',image)
        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        if rands[i].item() < p:
            m = "s"
            output = s_net(input_batch)
        elif rands[i].item() > p:
            m = "t"
            output = t_net(input_batch)
        print(f"i:{i}, m:{m}, fps:{fps}, rand:{rands[i]}, p:{p}, time:{time.time()-start}")
        i += 1
        # do something with output ...
        
        vals,idxs = torch.topk(output[0],5)
        vals = vals.softmax(dim=0)
        #top = list(enumerate(output[0].softmax(dim=0)))
        #vals, idxs = torch.topk()
        #top.sort(key=lambda x: x[1], reverse=True)
        #for idx, val in enumerate(vals):
        #    print(f"{val.item()*100:.2f}% {class_name_map[idxs[idx]]}            ")
        #print("\033[6A")

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            fps = frame_count / (now-last_logged)
            #print(f"{fps} fps")
            last_logged = now
            frame_count = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if i == 110:
             exit()
