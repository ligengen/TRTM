import numpy as np
import cv2

for i in range(4303):
    im1=cv2.imread('/Users/ligen/Desktop/cloth_recon/Depth_map/50/test/%05d.png0.png'%i, cv2.IMREAD_GRAYSCALE)
    im2=cv2.imread('/Users/ligen/Desktop/cloth_recon/Depth_map/50/pred_50/%05d_pred.png0.png'%i, cv2.IMREAD_GRAYSCALE)
    im3=cv2.imread('/Users/ligen/Desktop/cloth_recon/Depth_map/50/pred_152/%05d_pred.png0.png'%i,cv2.IMREAD_GRAYSCALE)
    # img_out=np.concatenate((im1,im2,im3),axis=1)
    # cv2.putText(img_out, 'ground truth', (300, 980), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1)
    # cv2.putText(img_out, 'resnet 50', (1400, 980), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1)
    # cv2.putText(img_out, 'resnet 152', (2500, 980), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1)
    # cv2.imwrite('/Users/ligen/Desktop/cloth_recon/Depth_map/50/merge/%05d.png'%i, img_out)
    im4=abs(np.array(im1)-np.array(im2))
    im5=abs(np.array(im1)-np.array(im3))
    im6=abs(np.array(im2)-np.array(im3))
    img_out=np.concatenate((im4,im5,im6),axis=1)
    # cv2.putText(img_out, '|GT-resnet50|', (300, 980), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1)
    # cv2.putText(img_out, '|GT-resnet152|', (1400, 980), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1)
    cv2.imwrite('/Users/ligen/Desktop/cloth_recon/Depth_map/50/difference/%05d.png'%i, img_out)
    if i == 3:
        import pdb
        pdb.set_trace()
