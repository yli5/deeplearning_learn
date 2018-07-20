import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import torchvision.models as models
from keras.preprocessing import image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from keras.applications.resnet50 import preprocess_input, decode_predictions
import glob
import time

# convert model to ONNX
use_cuda = torch.cuda.is_available()
dummy_input = Variable(torch.randn(1, 3, 224, 224))
save_model = models.alexnet(pretrained=True)
if use_cuda:
    dummy_input = dummy_input.cuda()
    save_model = save_model.cuda()
torch.onnx.export(save_model, dummy_input, "alexnet.proto")


# load model to measure accuracy
pytorch_model = models.alexnet(pretrained=True)

# load data
def process_input(url):
    img = image.load_img(url, target_size=(224, 224))
    return img

class MyCustomDataset(Dataset):
    def __init__(self, N, transforms=None):
        self.transforms = transforms
        self.count = N
        self.val = np.loadtxt('../dataset/val.txt', dtype=str)
        
    def __getitem__(self, index):
        data = process_input('../dataset/' + self.val[index][0])
        if self.transforms is not None:
            data = self.transforms(data)
        return ( data, int(self.val[index][1]) )

    def __len__(self):
        return self.count
       
 
N = 10
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([transforms.Scale(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
dataset = MyCustomDataset(N, transformations)

val_loader = torch.utils.data.DataLoader(dataset=dataset,
        batch_size=1, shuffle=False)


# compute accuracy
top1_correct_count = 0
top5_correct_count = 0
accum_time = 0.0
input_list = []
target_list = []
for i, (input, target) in enumerate(val_loader):
    input_list.append(input.data.numpy())
    target_list.append(target.data.numpy()[0])
    if use_cuda:
        target = target.cuda(async=True)
    input_var = Variable(input, volatile=True)
    target_var = Variable(target, volatile=True)
    
    start_ = time.time()
    output = pytorch_model(input_var)
    end_ = time.time()
    accum_time = accum_time + (end_ - start_)

    if target.cpu().data.numpy()[0] == output.argmax():
        top1_correct_count = top1_correct_count + 1
    if target.cpu().data.numpy()[0] in (-output).data.numpy().argsort()[0][:5]:
        top5_correct_count = top5_correct_count + 1

input_list = np.array(input_list)
target_list = np.array(target_list)
np.save('input_list', input_list)
np.savetxt('target_list.txt', target_list, fmt='%d')
print('top 1 accuracy : ', top1_correct_count * 1.0 / N)
print('top 5 accuracy : ', top5_correct_count * 1.0 / N)
print('throughput per sec : ', N * 1.0 / accum_time)

