#需要参考pix2pix中GAN的训练函数，及3d unet分割中的代码

import torch
from torch import nn
import os
from models.discriminator import *
from models.generator import *
from models.basic_block import *
from hsidataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
from tqdm.auto import tqdm
from metrics import PSNR, SSIM, SAM
from metrics_sklean import compute_hyper_psnr, compute_hyper_ssim


#创建Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import hsidataset
import h5py


    
#准备数据
DATA_HOME_DIR = "/mnt/liguanlin/DataSets/hypserdatasets/lowlight/"
train_data_dir = DATA_HOME_DIR + 'train/'
test_data_dir = DATA_HOME_DIR + 'test/'

#创建Dataset
train_dataset = HsiTrainDataset(train_data_dir)
#train_loader = DataLoader(dataset=train_set,  batch_size=batch_size, shuffle=True) 

#设置超参数
steps_per_epoch = 20
n_epochs=100
batch_size = 128
lr = 0.00001
device = DEVICE
display_step = 2
input_dim = 1
real_dim = 1


#创建模型及优化器
gen = UNetGenerator(input_dim)
gen = nn.DataParallel(gen).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(input_dim + real_dim)
disc = nn.DataParallel(disc).to(DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

"""
loss 函数的设计：
关于loss的设计的话。
也可以用三部分：
L1 loss，
光谱角loss

对抗生成网络的loss。BCEWithLogitsLoss
"""
# New parameters
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 200

def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial 
                  loss (which you aim to minimize)
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion 
                    loss (which you aim to minimize)
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''
    # Steps: 1) Generate the fake images, based on the conditions.
    #        2) Evaluate the fake images and the condition with the discriminator.
    #        3) Calculate the adversarial and reconstruction losses.
    #        4) Add the two losses, weighting the reconstruction loss appropriately.
    #### START CODE HERE ####
    fake = gen(condition)
    disc_fake_hat = disc(fake, condition)
    gen_adv_loss = adv_criterion(disc_fake_hat, torch.ones_like(disc_fake_hat))
    gen_rec_loss = recon_criterion(real, fake)
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss
    #### END CODE HERE ####
    return gen_loss

def val(val_loader, model, epoch, device):

    SAMs = []

    PSNRs_sk = []
    SSIMs_sk = []

    model.eval()

    val_psnr = 0
    count = 0

    with torch.no_grad():
        for condition, real in tqdm(val_loader):
            condition = condition.type(torch.FloatTensor)
            real = real.type(torch.FloatTensor)
        
            condition = condition.to(device)
            real = real.to(device)

            fake = model(condition)

            fake = fake.cpu().numpy().astype(np.float32)
            real = real.cpu().numpy().astype(np.float32)

            fake = np.squeeze(fake)
            real = np.squeeze(real)        

            sam = SAM(fake, real)
            SAMs.append(sam)

            psnr_sk = compute_hyper_psnr(real, fake)
            PSNRs_sk.append(psnr_sk)
            ssim_sk = compute_hyper_ssim(real, fake)
            SSIMs_sk.append(ssim_sk)
            count += 1
            print("===The {}-th picture sklearn =====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}".format(count,  psnr_sk, ssim_sk, sam))                 
    print("=====sklearn averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs_sk), np.mean(SSIMs_sk), np.mean(SAMs))) 
    




def train(save_model=False):
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    cur_step = 0

    val_dataset = HsiValDataset(test_data_dir)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False)    

    for epoch in range(n_epochs):

        gen.train()

        # Dataloader returns the batches
        for condition, real in tqdm(train_dataloader):
            
            condition = condition.type(torch.FloatTensor)
            real = real.type(torch.FloatTensor)
            
            condition = condition.to(DEVICE)
            real = real.to(DEVICE)

            ### Update discriminator ###
            disc_opt.zero_grad() #每次循环的时候，需清空之前跑结果时所得到的梯度信息
            with torch.no_grad():
                fake = gen(condition)
            
            disc_fake_hat = disc(fake.detach(), condition) # Detach generator
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))

            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # Update gradients
            disc_opt.step() # Update optimizer
            
            ### Update generator ###
            gen_opt.zero_grad() #zero out the gradient before backpropagation
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() #Update gradients
            gen_opt.step() #Update optimizer

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            #Logging
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, \
                        Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")

                mean_generator_loss = 0
                mean_discriminator_loss = 0

            #step ++,每一次循环，每一个batch的处理，叫做一个step
            cur_step += 1

        val(val_dataloader, gen, epoch, DEVICE)

        if save_model:
            torch.save({
                'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
            }, f"checkpoints/pix2pix3d_{epoch}.pth")


if __name__ == "__main__":
    train(save_model=True)
        
