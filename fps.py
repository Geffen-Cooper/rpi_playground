import time
import torch
import torchvision.models as models

# model = models.resnet34()
# img = torch.rand((3,224,224))
# img = img.unsqueeze(0)
# start = time.time()
# for i in range(100):
#     model(img)
# end = time.time()
# avg = (end-start)/100
# print("resnet32 (3x224x224) fps:",1/avg)

# model = models.mobilenet_v2()
torch.backends.quantized.engine = 'qnnpack'
model = models.quantization.mobilenet_v2(weights='DEFAULT',quantize=True)
model = torch.jit.script(model)
img = torch.rand((3,224,224))
img = img.unsqueeze(0)
start = time.time()
for i in range(2):
    model(img)
end = time.time()
avg2 = (end-start)/2
print("mobilenetv2 (3x64x64) fps:",1/avg2)
# print(avg/avg2)