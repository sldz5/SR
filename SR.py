import os,sys
import random
import argparse
import time
import datetime
import math

import torch ,visdom#
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from myssim import compare_ssim as ssim
from dataset import SRDataset
from model import Net,GANLoss,Discrim,SiameseNetwork
from tool import Visualizer

dataset_path = './'
temp_st = 5
count = 0
def tensor_to_PIL(input):
    """
    The function is to change tensor to PIL. 
    """
    input = input.squeeze(0).cpu().detach().numpy()
    max = input.max()
    input = input/max*255
    input = input.transpose(1,2,0).astype(np.uint8)
    output = Image.fromarray(input)
    return output

def _open_img(img):
    F = np.asarray(img).astype(float)/255.0
    return F

def _open_img_ssim(img):
    F = np.asarray(img)#.astype(float)
    return F

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=30):  ##https://discuss.pytorch.org/t/adaptive-learning-rate/320/26
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch == 0:
        lr_decay = lr_decay**(epoch / lr_decay_epoch)
        for param_group in optimizer.param_groups:
            if param_group['lr'] ==  1e-6:
                param_group['lr'] = 1e-6
            else :
                param_group['lr'] *= lr_decay
        print('lr is set to {}'.format(param_group['lr']))

def psnr_value(sr,gt):
    output = tensor_to_PIL(sr)
    groundt = tensor_to_PIL(gt)
    squared_error = np.square(_open_img(groundt) - _open_img(output))
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr,groundt,output

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-a","--action", nargs='?', choices=['train', 'valid', 'test','train_valid_test'], default='train_valid_test')
arg_parser.add_argument("-e", "--epoch", type=int, help="training epochs", default=2)
arg_parser.add_argument("-u", "--upscale_factor", type=int, help="upscale_factor", default=2)          
arg_parser.add_argument("-m", "--margin", type=float, help="training epochs", default=1.0)
arg_parser.add_argument("-c", "--cuda", action='store_true', default=False)
arg_parser.add_argument("-r", "--randaug", action='store_true', default=False)
arg_parser.add_argument("-l","--lr", type=float, default=1e-4,#5e-5
                    help='Learning Rate')
arg_parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate, default=0.0001')
arg_parser.add_argument("-w","--weight_decay", type=float, default=1.0e-8,  ##5.0e-7
                    help='weight decay')
arg_parser.add_argument("-b","--batch_size", type=int, default=5,
                    help='batch_size')
arg_parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
arg_parser.add_argument("-n","--nb_worker", type=int, default=4,
                    help='# of workers')
arg_parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
arg_parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
args = arg_parser.parse_args()


vis = Visualizer()
net = Net()
#netD = Discrim()
#siamese_net = SiameseNetwork(args.batch_size)
#model_file_s = "./models/b{}.159-e200.pkl".format(args.batch_size)  #保存siamese模型
#model_file = "./models/psnr27.49/P-b5.e10.pkl"         #保存net模型
#if model_file:                                                   
#    net.load_state_dict(torch.load(model_file,map_location='cpu')) 
#if model_file_s:
#    siamese_net.load_state_dict(torch.load(model_file_s))
#criterionGAN = GANLoss()
if args.cuda:
    net =net.cuda(0)
    #siamese_net = siamese_net.cuda(0)
    #criterionGAN = criterionGAN.cuda(0)
#else:
#    print ("cuda required.")
#    sys.exit(0)
##criterion = nn.CrossEntropyLoss()  
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,betas=(args.beta1, args.beta2))
#optimizer_s = torch.optim.Adam(siamese_net.parameters(),lr=args.lrD) #增加, momentum=0.9
#optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))  ##, betas=(opt.b1, opt.b2)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)
def train(args,epoch):
    # Train the Model
    net.train()
    
    exp_lr_scheduler(optimizer,epoch)
    #exp_lr_scheduler(optimizer_s,epoch)
    ds_train = SRDataset(dataset_path ,set='train')
    train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                               batch_size=args.batch_size,num_workers=args.nb_worker,shuffle=True)  # 将数据集的数据打乱
    temp_dataset = SRDataset(dataset_path,set = 'temp')
    temp_loader = torch.utils.data.DataLoader(dataset=temp_dataset,
                                               batch_size=1,num_workers=args.nb_worker,shuffle=False)
    print("Loaded {} train data.".format(len(train_loader)))
    total_loss   = []
    #total_loss_s = []
    for i, (images,gts,base_name) in enumerate(train_loader):   #,labels
        s1 = 0
        s2 = 0
        base_name1 , base_name2 = base_name
        with open('./temp.txt', 'r') as f:
            for line in f.readlines():        #txt中所有字符串读入data
                name=line.strip('\n')
                if base_name1 == name:
                    s1 = 1
                elif base_name2 == name:
                    s2 = 1
                else:
                    pass
        print("base_name:{}".format(base_name))
        if s1 == 1 and s2 == 0: #and string2 not in data:
            print("------1->0---------")
            images[0] = torch.Tensor(np.zeros([3,96,96]))
            gts[0] = torch.Tensor(np.zeros([3,96,96]))
        elif s2 == 1 and s1 == 0:
            print("------2->0---------")
            images[1] = torch.Tensor(np.zeros([3,96,96]))
            gts[1] = torch.Tensor(np.zeros([3,96,96]))
        elif s1 == 1 and s2 == 1:
            print("------in---------")
            continue  
        if args.cuda:
            images = Variable(images.cuda(0))
            gts    = Variable(gts.cuda(0))
        ##images = transform(images)
        ##gts = transform(gts) 
        """
        #-----------train siamese----------
        optimizer_s.zero_grad()
        #-----------false-----------
        #optimizer_D.zero_grad()
        outputs = net(images)
        pred_fake = siamese_net(outputs,gts)
        ture_lable = siamese_net(images,gts)
        if i % 100 == 0 :
            print("times:{}".format(i))
        fake_loss = criterionGAN(pred_fake,False)   
        real_loss = criterionGAN(ture_lable,True)
        loss_s = (fake_loss + real_loss)/2
        loss_s.backward()
        optimizer_s.step()
	    """
        #---------train net---------
        optimizer.zero_grad()
        outputs = net(images)
        
        #print("base_name:{}".format(base_name[:]))
        for j in range(0,len(outputs)):                        ####如果有psnr超过某个值，加进训练集
            psnr_sd,_,_ = psnr_value(images[j],gts[j])
            psnr_pdt,gt,sr = psnr_value(outputs[j],gts[j])
            temp = psnr_pdt - psnr_sd
            global temp_st
            global count
            if temp > 0.5 :
                if count == 3:
                    count = 0
                    temp_st = temp_st + 1
                count = count + 1
                #print("psnr_sd:{}".format(psnr_sd))
                #print("psnr_pdt:{}".format(psnr_pdt))
                print("temp:{}".format(temp))
                path_GT = "./temp_GT/"+base_name[j]+".png"
                path_LR = "./temp_LR/"+base_name[j]+".png"
                #output = output.resize([48,48],resample=Image.BICUBIC)
                gt.save(path_GT)
                sr.save(path_LR)
                k = 0
                #print("base_name[j]:{}".format(base_name[j]))
                with open('./temp.txt', 'r') as f:
                    for line in f.readlines():        #txt中所有字符串读入data
                        name=line.strip('\n')
                        #name = n.split("\\n")[0]
                        if base_name[j] == name:
                            k =  1
                if k == 1:
                    print("---------pass-------")
                    pass
                else :
                    f = open('./temp.txt','a')
                    f.write(base_name[j]+'\n')            ####不知道为什么，新写入的把旧的覆盖了，因为psnr更高了，因此不做修改，但和最初想法有违
                    f.close()
                     #print("~~~~string in data~~~~~")
        
        loss = criterion(outputs,gts)   ##loss_g
        #ture_lable = siamese_net(outputs,gts)
        #sia_loss = criterionGAN(ture_lable,True)
        ##turet_loss = criterionGAN(true_sia_t,True)
        #loss =(loss_g + sia_loss ) / 2
        #loss_t = loss_f+true_loss
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item()) ##data[0]
        #total_loss_s.append(loss_s.item())
    
    for i, (images,gts,base_name) in enumerate(temp_loader):   #,labels
        print("Loaded {} temp data.".format(len(temp_loader)))
        if args.cuda:
            images = Variable(images.cuda(0))
            gts    = Variable(gts.cuda(0))
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs,gts)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item()) ##data[0]    
    #aver_loss_s = np.mean(total_loss_s)
    aver_loss = np.mean(total_loss)
    

    print("T_aver_loss:{}".format(aver_loss))
    vis.plot_train_val(loss_train=aver_loss)
    #print('Epoch [%d/%d],s_Loss: %.8f f_Loss: %.8f' % (epoch+1, args.epoch, aver_loss_s,aver_loss)) ##
    print('Epoch [%d/%d],Loss: %.8f' % (epoch+1, args.epoch,aver_loss)) 

    
    if (epoch+1)%20==0:
        #model_file_s = "./models/b{}.{}-e{}.pkl".format(args.batch_size,epoch,args.epoch)  #保存siamese模型
        model_file = "./models/b{}.{}-e{}.pkl".format(args.batch_size,epoch,args.epoch)  #保存模型
        torch.save(net.state_dict(),model_file)                                  ##每迭代一次保存一下模型参数
        #torch.save(siamese_net.state_dict(), model_file_s)                         ####每迭代一次保存一下Siamese模型参数
    return net   ##siamese_net,

def valid(args, net=None):  ##siamese_net=None,
    net.eval()
    print("valid time is {} ".format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
    ##model_file_s = "./models/b{}.e{}.pkl".format(args.batch_size,args.epoch)  #保存siamese模型
    #model_file = "./models/b{}.e{}.pkl".format(args.batch_size,args.epoch)  #保存net模型
    ds_val = SRDataset(dataset_path, set='valid')
    valid_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                               batch_size=1,num_workers=args.nb_worker,shuffle=False) #
    ##if not siamese_net:
    ##    siamese_net = torch.load(model_file_s)                         
    if not net:
        net =torch.load(model_file)            
    
    
    print("Loaded {} valid data.".format(len(ds_val)))
    if args.cuda:
        #siamese_net = siamese_net.cuda(0)
        net =net.cuda(0)
   
    total_loss = []
    total_psnr = []
    total_ssim = []
    #num_epochs = args.epoch
    for i, (images,gts,base_name) in enumerate(valid_loader):  ##,labels
        if args.cuda:
            images = Variable(images.cuda(0))
            gts    = Variable(gts.cuda(0))
        outputs =net(images)
        
        loss_f = criterion(outputs,gts)
        total_loss.append(loss_f.item()) ##data[0]
        outputs = tensor_to_PIL(outputs)
        gts = tensor_to_PIL(gts)
        path = "./predict/"+base_name[0]+".png"
        outputs.save(path)

        squared_error = np.square(_open_img(gts) - _open_img(outputs))
        mse = np.mean(squared_error)
        psnr = 10 * np.log10(1.0 / mse)
        #print("{}psnr is {}".format(base_name[0],psnr))
        total_psnr.append(psnr)

        channels = []
        hr = _open_img_ssim(gts)
        sr = _open_img_ssim(outputs)
        for i in range(args.n_colors):
            channels.append(ssim(hr[:,:,i],sr[:,:,i], 
            gaussian_weights=True, use_sample_covariance=False))
        ssim_value = np.mean(channels)
        total_ssim.append(ssim_value)
        
    aver_loss =np.mean(total_loss)
    print("V_aver_loss:{}".format(aver_loss))   
    vis.plot_train_val(loss_val=aver_loss)

    aver_psnr = np.mean(total_psnr)
    aver_ssim = np.mean(total_ssim)
    print('PSNR:{}'.format(aver_psnr))
    print('SSIM:{}'.format(aver_ssim))

def test(args,net=None):
    #net.eval()           ###在验证集那里用过，不知道这里可不可以再用
    print("test time is {} ".format(time.strftime('%H:%M:%S',time.localtime(time.time()))))
    ##model_file_s = "./models/b{}.e{}.pkl".format(args.batch_size,args.epoch)  #保存siamese模型
    #model_file = "./models/b{}.e{}.pkl".format(args.batch_size,args.epoch)  #保存net模型
    model_file = "./models/aver_psnr27.4969-gan/P-b5.e10.pkl"
    #net.load_state_dict(torch.load(model_file,map_location='cpu'))
    ##if not siamese_net:
    ##    siamese_net = torch.load(model_file_s)                         
    if not net:
        net = Net()
        print("--------------")
        #net.load_state_dict(model_file)
        net.load_state_dict(torch.load(model_file,map_location='cpu'))            
    
    params = list(net.named_parameters())
    (name, param) = params[1]
    print(name)
    print(param.grad)
    print('-------------------------------------------------')
    (name2, param2) = params[2]
    print(name2)
    print(param2.grad)
    print('----------------------------------------------------')
    (name1, param1) = params[3]
    print(name1)
    print(param1.grad)


    if args.cuda:
        #siamese_net = siamese_net.cuda(0)
        net =net.cuda(0)
    
    total_psnr = []
    total_ssim = []
    #path_GT = './ValidationGT'
    path_LR = './Test_LR'                          ####test图片存放地址
    path_hazy = 'G:\\CVPR\\NTIRE2019\\crop_hazy' 
    transform = transforms.Compose([
                transforms.ToTensor(),
        ])
    print("Loaded {} test data.".format(len(os.listdir(path_LR))))
    
    for file in os.listdir(path_LR):
        starttime = datetime.datetime.now()
        

        #fullname_GT = path_GT + '/' +file
        fullname_LR = path_LR + '/' +file
        qianzui = file.split(".")[0]
        #GT = Image.open(fullname_GT)
        LR = Image.open(fullname_LR)
        LR = transform(LR).unsqueeze(0)  
        if args.cuda:
            LR = Variable(LR.cuda(0))
        print("LR:{}".format(LR.shape))
        SR =net(LR)
        SR = tensor_to_PIL(SR)
        save_path = "./test_predict/"+qianzui+".png"     ####保存生成图片的地址
        SR.save(save_path)
        endtime = datetime.datetime.now()
        
        print("one image time is : {}s ".format((endtime - starttime).seconds))
    """
        squared_error = np.square(_open_img(GT) - _open_img(SR))
        mse = np.mean(squared_error)
        psnr = 10 * np.log10(1.0 / mse)
        total_psnr.append(psnr)

        channels = []
        hr = _open_img_ssim(GT)
        sr = _open_img_ssim(SR)
        for i in range(args.n_colors):
            channels.append(ssim(hr[:,:,i],sr[:,:,i], 
            gaussian_weights=True, use_sample_covariance=False))
        ssim_value = np.mean(channels)
        total_ssim.append(ssim_value)
        print("{}-psnr:{},ssim:{}".format(qianzui,psnr,ssim_value))
    aver_psnr = np.mean(total_psnr)
    aver_ssim = np.mean(total_ssim)
    print('aver_PSNR:{}'.format(aver_psnr))
    print('aver_SSIM:{}'.format(aver_ssim))
    """

def main():
    print("Invoke {} with args {}".format(args.action, args))
    if args.action == "train":
        train(args)
    elif args.action == "valid":
        valid(args)
    elif args.action == "test":
        test(args)
    elif args.action == 'train_valid_test':
        #scheduler.step()
        num_epochs = args.epoch
        for epoch in range(num_epochs):
            net = train(args,epoch)   ##siamese_net,
            valid(args, net)  ##siamese_net,
        
        save_model(net,"./models/b{}.e{}.pkl".format(args.batch_size,args.epoch))
        #model_file="........"      ##上面这种保存方法的cpu加载方式
        #model.load_state_dict(torch.load(model_file))
        #torch.save(net, "./models/b{}.e{}.pkl".format(args.batch_size,args.epoch))             ###保存net模型
        torch.save(net.state_dict(),"./models/P-b{}.e{}.pkl".format(args.batch_size,args.epoch))  #保存net模型参数
        #torch.save(siamese_net, "./models/b{}.e{}.pkl".format(args.batch_size,args.epoch))                                
        print("Saved model at {}".format("./models"))
        test(args, net)


if __name__ == '__main__':
    #a = time.strftime('%H:%M:%S',time.localtime(time.time()))    #使用 time 模块的 strftime 方法来格式化日期
    starttime = datetime.datetime.now()

    print("start time is {} ".format(starttime))
    main()
    #b = time.strftime('%H:%M:%S',time.localtime(time.time()))
    endtime = datetime.datetime.now()
    print("Finish time is {} ".format(endtime))

    print("All time is {} ".format((endtime - starttime)))
    #shutdown = "/root/shutdown.sh"
    #r_v = os.system(shutdown)
