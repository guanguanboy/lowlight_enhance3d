from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def compute_hyper_psnr(label, noise):
    label_max_value = np.max(label)
    noise_max_value = np.max(noise)

    if label_max_value > noise_max_value:
        max_value = label_max_value
    else:
        max_value = noise_max_value 

    max_vlaue_int = np.ceil(max_value)

    psnr_value = psnr(label, noise, data_range=1)

    return psnr_value

def compute_hyper_ssim(label, noise):
    ssim_value = ssim(label, noise, multichannel=True)

    return ssim_value


