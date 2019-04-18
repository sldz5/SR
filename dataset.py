import os,sys
import numpy as np

import scipy.misc as m
from PIL import Image
from os import listdir
from os.path import join
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor,Compose,Resize,CenterCrop,Normalize

DATASET_PATH='./'
class SRDataset(Dataset):
    def __init__(self, root_path, set='train'):
        self.root_path = root_path
        self.set       = set
        
        self.transform = Compose([
                    ToTensor(),
            ])

        assert self.set in ['train', 'valid', 'trainval','temp']    #断言set在这三里面的其中一个


        self.files = []
        with open (self.root_path  + self.set + '.txt', 'r') as f:   #with 事先需要设置，事后做清理工作一步到位，详见https://www.cnblogs.com/DswCnblog/p/6126588.html
            for line in f:
                self.files.append(line.rstrip())    #rstrip()：删除 string 字符串末尾的指定字符（默认为空格）

        self.files = sorted(self.files)             #排序
        

    def __len__(self):
        return len(self.files)


    def _get_image(self, path):
        #img   = m.imread(path)
        y   = Image.open(path)#.convert('YCbCr')   ##转到特定像素空间
        #y, _,  _ =img.split()

        #npimg = np.array(img, dtype=np.uint8)

        # RGB => BGR
        #npimg = npimg[:, :, ::-1] # make a copy of the same list in reverse order:
        #npimg = npimg.astype(np.float64)                #转换类型

        ##npimg = m.imresize(npimg, (self.img_size, self.img_size))

        #npimg = npimg.astype(float) / 255.0

        #npimg = npimg.transpose(2,0,1) # （256，256，3）——> (3, 256, 256) 将下标（a[0],a[1],a[2]->a[2],a[0],a[1])

        #return torch.from_numpy(npimg).float()
        y = self.transform(y)
        return y

    def __getitem__(self, index):
        
        base_name = self.files[index]
        gt_file = self.root_path + self.set+'_GT/' + base_name + '.png'
        img_file  = self.root_path + self.set+'_LR/' + base_name + '.png'
        img = self._get_image(img_file)
        label=self._get_image(gt_file)

        
        return img ,label,base_name
  
"""
def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ =img.split()
    
    #y = np.array(y,dtype=np.uint8)
    #y = y[:,:,::-1]
    
    #y = y.astype(np.float64)
    #y = y.transpose(2,0,1)
    #y = torch.from_numpy(y).float()
    #y = y.unsqueeze(0)
    #print("y:{}".format(y.shape))
    return y

def calculate_valid_crop_size(crop_size, upscale_factor):   ###还不知道该函数干嘛的
    return crop_size - (crop_size % upscale_factor)

def input_transform(crop_size,upscale_factor):
    return Compose([
        Resize(crop_size),
        CenterCrop(crop_size),
        Resize(crop_size//upscale_factor),
        ToTensor(),
        Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

def target_transform(crop_size):
    return Compose([
        Resize(crop_size),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

class Imag_Dataset(Dataset):
    def __init__(self,image_dir,upscale_factor,set='train'):
        super(Dataset, self).__init__()
        self.image_dir = join(image_dir,set)
        self.image_filenames = [join(self.image_dir,x) for x in listdir(self.image_dir)]
        #self.set = set
        self.upscale_factor = upscale_factor

    def __getitem__(self,index):
        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        crop_size = calculate_valid_crop_size(256, self.upscale_factor)
        ##x , y = input_image.shape[1]//self.upscale_factor , input_image.shape[2]//self.upscale_factor
        #print("be_input_image:{}".format(input_image.size))
        input_image = input_transform(crop_size,self.upscale_factor)(input_image)
        #print("af_input_image:{}".format(input_image.shape))
        #input_image = input_image.unsqueeze(0)

        #print("after size:{}".format(input_image.shape))
        target = target_transform(crop_size)(target)
        
        #target = target.unsqueeze(0)

        #print("input_image:{},target:{}".format(input_image.shape,target.shape))
        return input_image, target

    def __len__(self):
        return len(self.image_filenames)
"""



### EOF ###
