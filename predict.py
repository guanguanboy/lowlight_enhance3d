
import torch
from torch import nn
import os
from models.discriminator import *
from models.generator import *
from models.basic_block import *
from hsidataset import *
from metrics import PSNR, SSIM, SAM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from tqdm.auto import tqdm

batch_size = 1
device = DEVICE
display_step = 2
input_dim = 1

def predict():
    
    #加载模型
    gen_model = UNetGenerator(input_dim).to(DEVICE)
    gen_model.load_state_dict(torch.load('./checkpoints/pix2pix3d_199.pth')['gen'])
    print('model loaded')

    #加载数据
    DATA_HOME_DIR = "/mnt/liguanlin/DataSets/hypserdatasets/lowlight/"
    train_data_dir = DATA_HOME_DIR + 'train/'
    
    #创建Dataset
    train_dataset = HsiTrainDataset(train_data_dir)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False)    

    #创建保存指标结果的列表
    PSNRs = []
    SSIMs = []
    SAMs = []
    count = 1
    #将每一个数据都使用模型去预测结果
    for condition, real in tqdm(dataloader):
        
        condition = condition.type(torch.FloatTensor)
        real = real.type(torch.FloatTensor)
        
        condition = condition.to(DEVICE)
        real = real.to(DEVICE)

        with torch.no_grad():
            fake = gen_model(condition)

        #计算各项指标
        fake = fake.cpu().numpy().astype(np.float32)
        real = real.cpu().numpy().astype(np.float32)

        psnr = PSNR(fake, real)
        #ssim = SSIM(fake, real)
        sam = SAM(fake, real)
        
        PSNRs.append(psnr)
        #SSIMs.append(ssim)
        SAMs.append(sam)

        #print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(count,  psnr, ssim, sam))                 
        print("===The {}-th picture=====PSNR:{:.3f}=====SAM:{:.3f}".format(count,  psnr, sam))                 
        
        count = count + 1

    print("=====averPSNR:{:.3f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SAMs))) 

if __name__=="__main__":
    predict()

