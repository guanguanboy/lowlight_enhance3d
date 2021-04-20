
import torch
from torch import nn
import os
from models.discriminator import *
from models.generator import *
from models.basic_block import *
from hsidataset import *
from metrics import PSNR, SSIM, SAM
import scipy.io as scio  
from metrics_sklean import compute_hyper_psnr, compute_hyper_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from tqdm.auto import tqdm

batch_size = 1
device = DEVICE
display_step = 2
input_dim = 1

def predict():
    
    #加载模型
    gen_model = UNetGenerator(input_dim)
    gen_model = nn.DataParallel(gen_model).to(DEVICE)
    gen_model.load_state_dict(torch.load('./checkpoints/pix2pix3d_199.pth')['gen'])
    print('model loaded')

    #加载数据
    DATA_HOME_DIR = "/mnt/liguanlin/DataSets/hypserdatasets/lowlight/"
    test_data_dir = DATA_HOME_DIR + 'test/'
    
    output_path = '/mnt/liguanlin/DataSets/hypserdatasets/lowlight/result/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #创建Dataset
    val_dataset = HsiValDataset(test_data_dir)
    dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False)    

    #创建保存指标结果的列表
    PSNRs = []
    SSIMs = []
    SAMs = []

    PSNRs_sk = []
    SSIMs_sk = []

    #fake_condition = torch.randn((1, 1, 224,384,384)).to(DEVICE)
    #fake1 = gen_model(fake_condition)
    #print('fake1.shape',fake1.shape)
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

        fake = np.squeeze(fake)
        real = np.squeeze(real)

        #print(fake.shape)
        #print(real.shape)

        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real)
        sam = SAM(fake, real)

        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)

        psnr_sk = compute_hyper_psnr(real, fake)
        PSNRs_sk.append(psnr_sk)
        ssim_sk = compute_hyper_ssim(real, fake)
        SSIMs_sk.append(ssim_sk)

        scio.savemat(output_path + str(count) + '.mat', {'enhanced':fake, 'label':real})
        print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}".format(count,  psnr, ssim, sam))                 
        #print("===The {}-th picture=====PSNR:{:.3f}=====SAM:{:.3f}".format(count,  psnr, sam))                 
        print("===The {}-th picture sklearn =====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}".format(count,  psnr_sk, ssim_sk, sam))                 

        if count == 5:
            break
        count = count + 1

    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs))) 
    print("=====sklearn averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs_sk), np.mean(SSIMs_sk), np.mean(SAMs))) 

if __name__=="__main__":
    predict()

