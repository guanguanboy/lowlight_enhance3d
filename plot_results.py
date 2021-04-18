import matplotlib.pyplot as plt
import scipy.io as scio  
import numpy as np

result_path = '/mnt/liguanlin/DataSets/hypserdatasets/lowlight/result/'
mat = scio.loadmat(result_path + '4.mat') 

enhanced = mat['enhanced']
label = mat['label']

enhanced = enhanced.transpose(1,2,0)
gray_img = enhanced[:,:,50]

print(gray_img.shape)
plt.imshow(gray_img)
plt.show()

label = label.transpose(1, 2, 0)
gray_img_label = label[:,:,50]
plt.imshow(gray_img_label.astype(np.int8))

plt.show()
#sudo_rgb = np.stack(enhanced[5,:,:], enhanced[25,:,:], enhanced[50,:,:])
#print(sudo_rgb.shape)
