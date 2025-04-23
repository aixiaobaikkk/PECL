"""
DICE->83.00 labeled

"""
import sys
import torch.nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb
from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch, Gridmask
from dataloaders.dataset import *
from net_factory import net_factory_3d
from utils.MCCP_utils import mix_loss, parameter_sharing, update_ema_variables, sup_loss,context_mask_PA
from utils.test_util import test_calculate_metric

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/Pancreas/new/', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MCCP_new', help='exp_name')
parser.add_argument('--model1', type=str, default='VNet', help='model1_name')
parser.add_argument('--model2', type=str, default='unet3d', help='model2_name')
parser.add_argument('--model3', type=str, default='voxresnet', help='model3_name')
parser.add_argument('--pre_max_iteration', type=int, default=9000, help='maximum pre-train iteration to train')  # 9000
parser.add_argument('--self_max_iteration', type=int, default=10000, help='maximum self-train iteration to train') # 9000
parser.add_argument('--max_samples', type=int, default=62, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.05, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=6, help='trained samples')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of MCCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()
def get_args():
    return args

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

#
patch_size = (96, 96, 96)
num_classes = 2


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def pre_train(args,model_name, snapshot_path):
    model = net_factory_3d(net_type=model_name, in_chns=1, class_num=num_classes, mode="train")
    db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):

            lr_ = base_lr * (1.0 - iter_num / self_max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_


            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][
                                                                                  :args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask_PA(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)


            grid_mask = Gridmask.GridMask_3D(prob=0.3)
            grid_mask.set_prob(iter_num, pre_max_iterations)
            volume_batch = grid_mask(volume_batch)

            if model_name == 'VNet':
                outputs, _ = model(volume_batch)
            else:
                outputs = model(volume_batch)

            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss = 0.3 * loss_ce + 0.7 * loss_dice

            iter_num += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            # if iter_num % 400 == 0 and iter_num > int(self_max_iterations/2):
            if epoch_num % 50 == 0 and epoch_num > max_epoch/2:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_PA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=16, stride_z=16)

                logging.info(model_name + " ==> " +
                             'iteration %d : loss: %03f, best_performence: %04f' % (iter_num, loss, best_dice))



                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(model_name))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                    logging.info("save best model to {}".format(save_mode_path))
                model.train()

    writer.close()


def self_train(args ,pre_snapshot_path1,pre_snapshot_path2, pre_snapshot_path3,self_snapshot_path):

    model1 = net_factory_3d(net_type=args.model1, in_chns=1, class_num=num_classes, mode="train")
    ema_model1 = net_factory_3d(net_type=args.model1, in_chns=1, class_num=num_classes, mode="train")
    model2 = net_factory_3d(net_type=args.model2, in_chns=1, class_num=num_classes, mode="train")
    ema_model2 = net_factory_3d(net_type=args.model2, in_chns=1, class_num=num_classes, mode="train")
    model3 = net_factory_3d(net_type=args.model3, in_chns=1, class_num=num_classes, mode="train")
    ema_model3 = net_factory_3d(net_type=args.model3, in_chns=1, class_num=num_classes, mode="train")
    # ema_model set
    for param in ema_model1.parameters():
        param.detach_()
    for param in ema_model2.parameters():
        param.detach_()
    for param in ema_model3.parameters():
        param.detach_()
    db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum
    print(labelnum)
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer3 = optim.SGD(model3.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model1 = os.path.join(pre_snapshot_path1, f'{args.model1}_best_model.pth')
    pretrained_model2 = os.path.join(pre_snapshot_path2, f'{args.model2}_best_model.pth')
    pretrained_model3 = os.path.join(pre_snapshot_path3, f'{args.model3}_best_model.pth')

    load_net(model1, pretrained_model1)
    load_net(ema_model1, pretrained_model1)
    load_net(model2, pretrained_model2)
    load_net(ema_model2, pretrained_model2)
    load_net(model3, pretrained_model3)
    load_net(ema_model3, pretrained_model3)

    model1.train()
    ema_model1.train()
    model2.train()
    ema_model2.train()
    model3.train()
    ema_model3.train()

    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader)
    loss_mse = torch.nn.MSELoss()
    for epoch in tqdm(range(max_epoch), ncols=70):
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img = volume_batch[:args.labeled_bs]
            img_lab = label_batch[:args.labeled_bs]

            unimg = volume_batch[args.labeled_bs:]

            # 获得可靠伪标签：
            """
            model1 ==> VNet
            model2 ==> UNet3D
            model3 ==> VoxResUNet
            """
            with torch.no_grad():
                vnet_out, _ = ema_model1(unimg)
                unet3d_out = ema_model2(unimg)
                voxresunet_out = ema_model3(unimg)

                vnet_out_soft = F.softmax(vnet_out, dim=1)
                unet3d_out_soft = F.softmax(unet3d_out, dim=1)
                voxresunet_out_soft = F.softmax(voxresunet_out, dim=1)

                vnet_predict = torch.max(vnet_out_soft[:args.batch_size, :, :, :, :], 1, )[1]
                unet3d_predict = torch.max(unet3d_out_soft[:args.batch_size, :, :, :, :], 1, )[1]
                voxresunet_predict = torch.max(voxresunet_out_soft[:args.batch_size, :, :, :, :], 1, )[1]

                diff_mask_vnet_unet3d = ((unet3d_predict == 1) ^ (vnet_predict == 1)).to(torch.int32).cuda()
                diff_mask_vnet_voxresunet = ((vnet_predict == 1) ^ (voxresunet_predict == 1)).to(torch.int32).cuda()

                vnet_vote_vnet_unet3d = diff_mask_vnet_unet3d * vnet_predict
                unet3d_vote_vnet_unet3d = diff_mask_vnet_unet3d * unet3d_predict
                voxresunet_vote_vnet_unet3d = diff_mask_vnet_unet3d * voxresunet_predict
                vnet_vote_vnet_voxresunet = diff_mask_vnet_voxresunet * vnet_predict
                unet3d_vote_vnet_voxresunet = diff_mask_vnet_voxresunet * unet3d_predict
                voxresunet_vote_vnet_voxresunet = diff_mask_vnet_voxresunet * voxresunet_predict

                all_predictions = torch.stack(
                    [vnet_vote_vnet_unet3d, unet3d_vote_vnet_unet3d, voxresunet_vote_vnet_unet3d], dim=4)

                # 找到每个位置最常见的预测值
                vote_prediction_unet3d, _ = torch.mode(all_predictions, dim=4)
                all_predictions = torch.stack(
                    [vnet_vote_vnet_voxresunet, unet3d_vote_vnet_voxresunet, voxresunet_vote_vnet_voxresunet], dim=4)
                # 找到每个位置最常见的预测值
                vote_prediction_voxresunet, _ = torch.mode(all_predictions, dim=4)
                unimg_lab = vnet_predict * (1 - diff_mask_vnet_unet3d) + vote_prediction_unet3d * diff_mask_vnet_unet3d
                unimg_lab = unimg_lab * (
                            1 - diff_mask_vnet_voxresunet) + vote_prediction_voxresunet * diff_mask_vnet_voxresunet

                img_mask, loss_mask = context_mask_PA(img, args.mask_ratio)

            mixl_img = img * img_mask + unimg * (1 - img_mask)
            mixu_img = img * (1 - img_mask) + unimg * img_mask

            mixu_lab = img_lab * (1 - img_mask) + unimg_lab * img_mask
            mixl_lab = img_lab * img_mask + unimg_lab * (1 - img_mask)


            outputs_l, _ = model1(mixl_img)
            outputs_u, _ = model1(mixu_img)



            loss_l = sup_loss(outputs_l, mixl_lab)
            loss_u = sup_loss(outputs_u, mixu_lab)


            loss_mix_lab = mix_loss(outputs_l, img_lab, vnet_predict, loss_mask, u_weight=args.u_weight)
            loss_mix_unl = mix_loss(outputs_u, vnet_predict, img_lab, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_mix = 2 * (loss_mix_lab + loss_mix_unl)
            loss = loss_l + loss_u + loss_mix


            iter_num += 1
            writer.add_scalar('Self/loss_l', loss_l, iter_num)
            writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)
            loss12 = 2 * loss_mse(vnet_predict.float(), unet3d_predict.float())# * get_current_consistency_weight(
               #iter_num)
            loss13 = 2 * loss_mse(vnet_predict.float(), voxresunet_predict.float())# * get_current_consistency_weight(
               #iter_num)



            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            optimizer2.zero_grad()
            loss12.requires_grad_(True)
            loss12.backward()
            optimizer2.step()
            optimizer3.zero_grad()
            loss13.requires_grad_(True)
            loss13.backward()
            optimizer3.step()


            iter_num += 1


            update_ema_variables(model1, ema_model1, 0.99)
            update_ema_variables(model2, ema_model2, 0.99)
            update_ema_variables(model3, ema_model3, 0.99)
            lr_ = base_lr * (1.0 - epoch / max_epoch) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            for param_group in optimizer3.param_groups:
                param_group['lr'] = lr_


        if epoch % 50 == 0 and epoch > max_epoch/2:
            model1.eval()
            dice_sample = test_3d_patch.var_all_case_PA(model1, num_classes=num_classes, patch_size=patch_size,
                                                        stride_xy=16, stride_z=16)
            if dice_sample > best_dice:
                best_dice = round(dice_sample, 4)
                save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                save_best_path = os.path.join(self_snapshot_path, '{}_best_model.pth'.format(args.model1))
                # save_net_opt(model, optimizer, save_mode_path)
                # save_net_opt(model, optimizer, save_best_path)
                torch.save(model1.state_dict(), save_mode_path)
                torch.save(model1.state_dict(), save_best_path)
                logging.info("save best model to {}".format(save_mode_path))
            writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
            writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
            model1.train()

        logging.info('epochs %d : loss: %03f  loss_l: %03f  loss_u: %03f  loss_mix: %03f'
                     'loss12: %03f  loss13: %03f best_dice : %04f' % (epoch,
                                                                      loss, loss_l, loss_u, loss_mix, loss12,
                                                                      loss13, best_dice))
    writer.close()


def run_main(exp, labelnum):
    args.labelnum = labelnum

    ## make logger file
    pre_snapshot_path1 = "./model/MCCP_1030/PA_{}_{}_labeled/exp_{}/pre_train_model1".format(args.exp, args.labelnum, exp)
    pre_snapshot_path2 = "./model/MCCP_1030/PA_{}_{}_labeled/exp_{}/pre_train_model2".format(args.exp, args.labelnum, exp)
    pre_snapshot_path3 = "./model/MCCP_1030/PA_{}_{}_labeled/exp_{}/pre_train_model3".format(args.exp, args.labelnum, exp)
    self_snapshot_path = "./model/MCCP_1030/PA_{}_{}_labeled/exp_{}/self_train".format(args.exp, args.labelnum, exp)
    print("Strating MCCP training.")
    for snapshot_path in [pre_snapshot_path1, pre_snapshot_path2, pre_snapshot_path3, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    # -- Pre-Training
    logging.basicConfig(filename=pre_snapshot_path1+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("====================================================================================")
    pre_train(args, args.model1, pre_snapshot_path1)
    logging.info("====================================================================================")
    pre_train(args, args.model2, pre_snapshot_path2)
    logging.info("====================================================================================")
    pre_train(args, args.model3, pre_snapshot_path3)

    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path1,pre_snapshot_path2, pre_snapshot_path3,self_snapshot_path)




if __name__ == "__main__":


    for exp in range(1000,1001):
        run_main(exp, 12)
        run_main(exp, 6)
    # run_main(0, 12)
    # run_main(1, 12)
    # run_main(2, 12)








