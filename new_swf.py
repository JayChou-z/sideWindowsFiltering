import cv2
import numpy

import torch
import torch.nn as nn

import numpy as np



class sideWinFiltering(nn.Module):
    def __init__(self, radius, iteration):
        super(sideWinFiltering, self).__init__()
        self.radius = radius
        self.iteration = iteration
        self.kernel_size = 2 * self.radius + 1
        self.convL = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                               kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convR = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                               kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convU = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                               kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convD = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                               kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convNW = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                                kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convNE = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                                kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convSW = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                                kernel_size=(self.kernel_size, self.kernel_size), bias=False)
        self.convSE = nn.Conv2d(1, 1, padding=self.radius, padding_mode='replicate',
                                kernel_size=(self.kernel_size, self.kernel_size), bias=False)

    def forward(self, img):
        b, c, h, w = img.size()
        d = torch.zeros((b, 8, h, w), dtype=torch.float32)
        res = img.clone()
        filter = torch.Tensor(torch.ones((1, 1, self.kernel_size, self.kernel_size),dtype=torch.float32))

        L, R, U, D = [filter.clone() for _ in range(4)]

        L[:, :, :, self.radius + 1:] = 0
        R[:, :, :, 0: self.radius] = 0
        U[:, :, self.radius + 1:, :] = 0
        D[:, :, 0: self.radius, :] = 0

        NW, NE, SW, SE = U.clone(), U.clone(), D.clone(), D.clone()

        L, R, U, D = L / ((self.radius + 1) * self.kernel_size), R / ((self.radius + 1) * self.kernel_size), \
                     U / ((self.radius + 1) * self.kernel_size), D / ((self.radius + 1) * self.kernel_size)

        NW[:, :, :, self.radius + 1:] = 0
        NE[:, :, :, 0:self.radius] = 0
        SW[:, :, :, self.radius + 1:] = 0
        SE[:, :, :, 0:self.radius] = 0

        NW, NE, SW, SE = NW / ((self.radius + 1) ** 2), NE / ((self.radius + 1) ** 2), \
                         SW / ((self.radius + 1) ** 2), SE / ((self.radius + 1) ** 2)


        for ch in range(c):
            im_ch = img[:, ch, :, :].clone().view(b, 1, h, w)

            for i in range(self.iteration):
                self.convL.weight.data = L
                self.convR.weight.data = R
                self.convU.weight.data = U
                self.convD.weight.data = D
                self.convNW.weight.data = NW
                self.convNE.weight.data = NE
                self.convSW.weight.data = SW
                self.convSE.weight.data = SE

                d[:, 0, :, :] = self.convL(im_ch) - im_ch
                d[:, 1, :, :] = self.convR(im_ch) - im_ch
                d[:, 2, :, :] = self.convU(im_ch) - im_ch
                d[:, 3, :, :] = self.convD(im_ch) - im_ch
                d[:, 4, :, :] = self.convNW(im_ch) - im_ch
                d[:, 5, :, :] = self.convNE(im_ch) - im_ch
                d[:, 6, :, :] = self.convSW(im_ch) - im_ch
                d[:, 7, :, :] = self.convSE(im_ch) - im_ch


                d_abs = torch.abs(d)
                mask_min = torch.argmin(d_abs, dim=1, keepdim=True)
                dm = torch.gather(input=d, dim=1, index=mask_min)  #很对
                im_ch = dm + im_ch
            res[:, ch, :, :] = im_ch
        return res



img=cv2.imread('0019.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
radius = 3
iteration = 10
img1=img.astype(np.float32)

imgt=np.transpose(img1,(2,0,1))[None,:,:,:]
img2 = torch.tensor(imgt, dtype=torch.float32)

s = sideWinFiltering(radius, iteration)
res = s.forward(img2)
res=img2-res

if res.size(1) == 3:
    img_res1 = np.transpose(np.squeeze(res.data.numpy()), (1, 2, 0))
else:
    img_res1 = np.squeeze(res.data.numpy())


img_res1=np.clip(img_res1,0,255)
winname = 'window'

img_res2 = img_res1.astype(np.uint8)

cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
cv2.imshow(winname,img_res2)
cv2.waitKey(0)


