from networks.Two_demention.CNN_based.unet import UNet_2d
from networks.Two_demention.CNN_based.att_unet import AttU_Net
from networks.Two_demention.Transformer_based_CASCADE.networks import PVT_CASCADE
from networks.Three_demention.CNN_based.unet_3D import unet_3D
from networks.Three_demention.CNN_based.VNet import VNet
from networks.Three_demention.CNN_based.VoxResNet import VoxResNet
from networks.tri_net.LUCF_Net import LUCF_Net

def net_factory_3d(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "VNet" and mode == "train" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    if net_type == "unet3d" and mode == "train" and tsne == 0:
        net = unet_3D(feature_scale=4, n_classes=class_num, is_deconv=True, in_channels=in_chns, is_batchnorm=True).cuda()
    if net_type == "unet3d" and mode == "test" and tsne == 0:
        net = unet_3D(feature_scale=4, n_classes=class_num, is_deconv=True, in_channels=in_chns, is_batchnorm=True).cuda()
    if net_type == "voxresnet" and mode == "train" and tsne == 0:
        net = VoxResNet(in_chns=in_chns,class_num=class_num,feature_chns=64).cuda()
    if net_type == "voxresnet" and mode == "test" and tsne == 0:
        net = VoxResNet(in_chns=in_chns,class_num=class_num,feature_chns=64).cuda()
    return net


def net_factory_2d(net_type="unet", in_chns=1, class_num=2, mode="train", ema=False):
    if net_type == "unet" and mode == "train":
        net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()

    if net_type == "att_unet":
        net = AttU_Net(img_ch=in_chns,output_ch=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()
    if net_type == "pvt_cascade":
        net = PVT_CASCADE(n_class=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()

    if net_type == "LUCFNet":
        net = LUCF_Net(in_chns=in_chns, class_num=class_num).cuda()
        if ema:
            for param in net.parameters():
                param.detach_()

    return net




