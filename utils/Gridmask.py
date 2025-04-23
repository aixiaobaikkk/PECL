import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math
import numpy as np
from scipy.ndimage import rotate
# 导入用于3D数据处理的库，例如PyTorch 3D
import torchvision.transforms.functional as F3D

import numpy as np
import torch
import math
from PIL import Image

class Grid_3D(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        # self.prob = self.st_prob * min(1, epoch / max_epoch)
        # print(epoch, "epoch")
        # print(max_epoch, "max_epoch")

        self.prob = self.st_prob * min(1, epoch / max_epoch)
        # print(self.prob, "prob")

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        d_img = img.size(3)

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w + d_img * d_img)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh, hh), np.float32)
        # print(mask.shape,"1")
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        st_d = np.random.randint(d)

        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :, :] *= 0

        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t, :] *= 0

        for i in range(-1, hh // d + 1):
            s = d * i + st_d
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, :, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = rotate(mask, r, axes=(0, 1), reshape=False)
        # mask = Image.fromarray(np.uint8(mask))
        # mask = mask.rotate(r)
        # mask = np.asarray(mask)
        # print(mask.shape,'2')
        # print()
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w, (hh - d_img) // 2 :(hh - d_img) // 2 + d_img]
        # print(mask.shape, '3')
        # mask = torch.from_numpy(mask).float().cuda()
        mask = torch.from_numpy(mask.copy()).float().cuda()

        # mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GridMask_3D(nn.Module):
    def __init__(self, d1=16, d2=32, rotate=360, ratio=0.4, mode=1, prob=0.2):
        super(GridMask_3D, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid_3D(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        # if not self.training:
        #     return x

        n, c, h, w,d = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))

        y = torch.cat(y).view(n, c, h, w, d)

        return y

class Grid_2D(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        # self.prob = self.st_prob * min(1, epoch / max_epoch)
        # print(epoch, "epoch")
        # print(max_epoch, "max_epoch")

        self.prob = self.st_prob * min(1, epoch / max_epoch)
        # print(self.prob, "prob")

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0

        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        # mask = torch.from_numpy(mask).float().cuda()
        mask = torch.from_numpy(mask.copy()).float().cuda()

        # mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GridMask_2D(nn.Module):
    def __init__(self, d1=96, d2=224, rotate=360, ratio=0.4, mode=1, prob=0.8):
        super(GridMask_2D, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid_2D(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))

        y = torch.cat(y).view(n, c, h, w)
        # print("win")

        return y

if __name__ == "__main__":
    import numpy as np
    import torch
    from mayavi import mlab
    import nrrd
    from torchvision.transforms import ToTensor


    # Load nrrd file
    filename = r'F:\segment dataset\2018_UTAH_MICCAI\Training Set\0RZDK210BSMWAA6467LU\lgemri.nrrd'
    volume, header = nrrd.read(filename)

    # Ensure volume is float32 and reshape if needed
    volume = volume.astype(np.float32)
    if len(volume.shape) == 4:
        volume = np.transpose(volume, (3, 2, 1, 0))  # Adjust axes if necessary

    # Normalize volume data to [0, 1]
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    # Convert volume to PyTorch tensor and move to GPU if available
    volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float().cuda()

    # Create a GridMask3D object and set parameters
    gridmask = GridMask_3D(rotate=360, ratio=0.4, mode=1, prob=0.8)
    gridmask.eval()  # Switch to evaluation mode to avoid randomness

    # Apply GridMask augmentation
    augmented_volume_tensor = gridmask(volume_tensor)

    # Convert PyTorch tensor back to NumPy array for visualization
    augmented_volume = augmented_volume_tensor.squeeze().cpu().numpy()

    # Visualize original volume data
    mlab.figure(size=(800, 800))
    mlab.contour3d(volume.squeeze(), contours=2, opacity=0.5)
    mlab.colorbar(title='Original Volume')
    mlab.show()

    # Visualize augmented volume data
    mlab.figure(size=(800, 800))
    mlab.contour3d(augmented_volume.squeeze(), contours=2, opacity=0.5)
    mlab.colorbar(title='Augmented Volume (GridMask)')
    mlab.show()




































# # 创建Mayavi场景
# mlab.figure(size=(800, 800))
#
# # 使用contour3d绘制体素
# contour = mlab.contour3d(volume.squeeze(), contours=8, opacity=0.5, colormap='autumn')
#
# # 获取颜色映射
# lut_manager = contour.module_manager.scalar_lut_manager
#
# # 获取颜色映射对象
# lut = lut_manager.lut
#
# # 将0数值的体素颜色设置为黑色
# # lut.SetTableRange(1, augmented_volume.max())  # 将颜色映射范围设置为不包括0的范围
# # lut.SetTableValue(0, 0, 0, 0)  # 将0数值的颜色设置为黑色
#
# # 添加颜色条
# mlab.colorbar(title='Augmented Volume (GridMask)')
#
# # 显示Mayavi场景
# mlab.show()
#
#
# # 创建Mayavi场景
# mlab.figure(size=(800, 800))
#
# # 使用contour3d绘制体素
# contour = mlab.contour3d(augmented_volume.squeeze(), contours=8, opacity=0.5, colormap='autumn')
#
# # 获取颜色映射
# lut_manager = contour.module_manager.scalar_lut_manager
#
# # 获取颜色映射对象
# lut = lut_manager.lut
#
# # 将0数值的体素颜色设置为黑色
# # lut.SetTableRange(1, augmented_volume.max())  # 将颜色映射范围设置为不包括0的范围
# # lut.SetTableValue(0, 0, 0, 0)  # 将0数值的颜色设置为黑色
#
# # 添加颜色条
# mlab.colorbar(title='Augmented Volume (GridMask)')
#
# # 显示Mayavi场景
# mlab.show()
