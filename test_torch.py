import torch
from torchvision import models
import time

torch.backends.quantized.engine = 'qnnpack'
model = models.shufflenet_v2_x0_5().eval()
model = models.quantization.mobilenet_v2(quantize=True)

#model = torch.jit.script(model)
torch.set_num_threads(2)
rand = torch.rand(1,3,224,224)

start = time.time()
frame_count = 0

while True:
	with torch.no_grad():
		model(rand)
		frame_count += 1
		curr_time = time.time()
		elapsed = curr_time - start
		if elapsed >= 1:
			print(f"FPS: {round(frame_count/elapsed,1)}")
			start = curr_time
			frame_count = 0
