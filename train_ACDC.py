"""
todo : 90 dsc
"""

import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms

from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from net_factory import net_factory_2d
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d, Gridmask
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Lr0.15', help='experiment_name')
parser.add_argument('--model1', type=str, default='unet', help='model1_name')
parser.add_argument('--model2', type=str, default='att_unet', help='model2_name')
parser.add_argument('--model3', type=str, default='pvt_cascade', help='model3_name')

parser.add_argument('--pre_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=9000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.15, help='segmentation network learning rate')#7、14用0.1
parser.add_argument('--self_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--grid_ratio', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--workers', type=int, default=0, help='')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--consistency', type=float, default=10, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=5000.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
args = parser.parse_args()
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def pre_train(args, model_name, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    model = net_factory_2d(net_type=model_name, in_chns=1, class_num=num_classes, mode="train")
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=args.workers ,pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.workers)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)
            grid_mask = Gridmask.GridMask_2D(prob=args.grid_ratio)
            grid_mask.set_prob(iter_num, max_iterations)
            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            net_input = grid_mask(net_input)

            out_mixl = model(net_input)
            loss = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            # writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            # writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info(model_name + ' ==> ' + 'iteration %d loss: %f best_performence: %4f' % (
            iter_num, loss, best_performance))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num % 200 == 0 and iter_num > int(max_iterations/ 2):
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(model_name))
                    # save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args ,pre_snapshot_path1,pre_snapshot_path2, pre_snapshot_path3,self_snapshot_path):
    from test_ACDC import Inference_test
    test_performence = 0.0
    base_lr = args.self_lr #学习率？？？？
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model1 = os.path.join(pre_snapshot_path1, '{}_best_model.pth'.format(args.model1))
    pre_trained_model2 = os.path.join(pre_snapshot_path2, '{}_best_model.pth'.format(args.model2))
    pre_trained_model3 = os.path.join(pre_snapshot_path3, '{}_best_model.pth'.format(args.model3))



    model1 = net_factory_2d(net_type=args.model1,in_chns=1, class_num=num_classes)
    ema_model1 = net_factory_2d(net_type=args.model1,in_chns=1, class_num=num_classes,ema=True)
    model2 = net_factory_2d(net_type=args.model2, in_chns=1, class_num=num_classes)
    ema_model2 = net_factory_2d(net_type=args.model2, in_chns=1, class_num=num_classes, ema=True)
    model3 = net_factory_2d(net_type=args.model3, in_chns=1, class_num=num_classes)
    ema_model3 = net_factory_2d(net_type=args.model3, in_chns=1, class_num=num_classes, ema=True)



    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.workers)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer3 = optim.SGD(model3.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    load_net(ema_model1, pre_trained_model1)
    load_net_opt(model1, optimizer1, pre_trained_model1)

    load_net(ema_model2, pre_trained_model2)
    load_net_opt(model2, optimizer2, pre_trained_model2)

    load_net(ema_model3, pre_trained_model3)
    load_net_opt(model3, optimizer3, pre_trained_model3)

    logging.info("Loaded from {}".format(pre_trained_model1))
    logging.info("Loaded from {}".format(pre_trained_model2))
    logging.info("Loaded from {}".format(pre_trained_model3))

    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model1.train()
    ema_model1.train()

    model2.train()
    ema_model2.train()

    model3.train()
    ema_model3.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    loss_mse = torch.nn.MSELoss(reduction='mean')

    test_best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img = volume_batch[:args.labeled_bs]
            unimg = volume_batch[args.labeled_bs:]
            img_lab = label_batch[:args.labeled_bs]
            with torch.no_grad():
                model1_out = F.softmax(ema_model1(unimg), dim=1)
                model2_out = F.softmax(ema_model2(unimg), dim=1)
                model3_out = F.softmax(ema_model3(unimg), dim=1)

                model1_predict = torch.argmax(model1_out, dim=1)
                model2_predict = torch.argmax(model2_out, dim=1)
                model3_predict = torch.argmax(model3_out, dim=1)

                # 由两个网络产生的伪标签1和伪标签2，对于伪标签上的每个像素点，产生一个和伪标签一样的mask，该mask的每个像素点的赋值规则为：若两个伪标签的预测一致则让该点的数值为0，若不一致则为1
                mask_model1_model2 = torch.zeros_like(model1_predict)
                mask_model1_model2[model1_predict != model2_predict] = 1
                mask_model1_model2[model1_predict == model2_predict] = 0

                mask_model1_model3 = torch.zeros_like(model1_predict)
                mask_model1_model3[model1_predict != model3_predict] = 1
                mask_model1_model3[model1_predict == model3_predict] = 0

                model1_vote_diff12 = model1_predict * mask_model1_model2
                model2_vote_diff12 = model2_predict * mask_model1_model2
                model3_vote_diff12 = model3_predict * mask_model1_model2

                model1_vote_diff13 = model1_predict * mask_model1_model3
                model2_vote_diff13 = model2_predict * mask_model1_model3
                model3_vote_diff13 = model3_predict * mask_model1_model3

                all_predictions_diff12 = torch.stack([model1_vote_diff12, model2_vote_diff12, model3_vote_diff12])
                # 找到每个像素点最常见的值
                most_common_values_diff12, _ = torch.mode(all_predictions_diff12, dim=0)

                all_predictions_diff13 = torch.stack([model1_vote_diff13, model2_vote_diff13, model3_vote_diff13])
                # 找到每个像素点最常见的值
                most_common_values_diff13, _ = torch.mode(all_predictions_diff13, dim=0)

                unimg_lab = model1_predict * (1 - mask_model1_model2) + most_common_values_diff12
                unimg_lab = unimg_lab * (1 - mask_model1_model3) + most_common_values_diff13

                img_mask, loss_mask = generate_mask(img)



            mixl_img = img * img_mask + unimg * (1 - img_mask)
            mixu_img = img * (1 - img_mask) + unimg * img_mask


            mixl_lab = img_lab * img_mask + unimg_lab * (1 - img_mask)
            mixu_lab = img_lab * (1 - img_mask) + unimg_lab * img_mask


            # grid_mask = Gridmask.GridMask_2D()
            # print(iterator, max_iterations, "here")
            # grid_mask.set_prob(iter_num, max_iterations)
            #
            # mixu_img = grid_mask(mixu_img)
            # mixl_img = grid_mask(mixl_img)

            out_l = model1(mixl_img)
            out_unl = model1(mixu_img)

            loss_l = sup_loss(out_l, mixl_lab)
            loss_u = sup_loss(out_unl, mixu_lab)

            loss_mix = mix_loss(out_l, img_lab, model1_predict, loss_mask, u_weight=args.u_weight) \
                       + mix_loss(out_unl, model1_predict, img_lab, loss_mask, u_weight=args.u_weight, unlab=True)

            loss = loss_l + loss_u + 4 * loss_mix
            loss12 = loss_mse(model1_predict.float(), model2_predict.float()) * get_current_consistency_weight(iter_num)
            loss13 = loss_mse(model1_predict.float(), model3_predict.float()) * get_current_consistency_weight(iter_num)
            # loss12 = loss_mse(model1_predict.float(), model2_predict.float())
            # loss13 = loss_mse(model1_predict.float(), model3_predict.float())


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
            update_model_ema(model1, ema_model1, 0.99)
            update_model_ema(model2, ema_model2, 0.99)
            update_model_ema(model3, ema_model3, 0.99)
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # for param_group in optimizer1.param_groups:
            #     param_group['lr'] = lr_
            logging.info(
                'iteration %d: loss: %f, mix_loss: %f, sup_loss: %f, loss12: %f, loss13: %f, best_performence: %f' % (
                    iter_num, loss, loss_mix, loss_l + loss_u, loss12, loss13, best_performance))
            if iter_num > 0 and iter_num % 200 == 0:

                model1.eval()
                metric_list1 = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i_1 = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list1 += np.array(metric_i_1)
                metric_list1 = metric_list1 / len(db_val)
                performance1 = np.mean(metric_list1, axis=0)[0]

                if performance1 > best_performance:
                    best_performance = performance1
                    save_mode_path = os.path.join(self_snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model1))
                    # torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best_path)
                model1.train()


                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance1))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def run_main(exp,labelnum):


    args.labelnum = labelnum

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path1 = "./model/{}/ACDC_{}_labeled/exp_{}/pre_train_model1".format(args.exp, args.labelnum,exp)
    pre_snapshot_path2 = "./model/{}/ACDC_{}_labeled/exp_{}/pre_train_model2".format(args.exp, args.labelnum,exp)
    pre_snapshot_path3 = "./model/{}/ACDC_{}_labeled/exp_{}/pre_train_model3".format(args.exp, args.labelnum,exp)
    self_snapshot_path = "./model/{}/ACDC_{}_labeled/exp_{}/self_train".format(args.exp, args.labelnum, exp)

    for snapshot_path in [pre_snapshot_path1,pre_snapshot_path2, pre_snapshot_path3,self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

    # Pre_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("====================================================================================")
    pre_train(args, args.model1, pre_snapshot_path1)
    logging.info("====================================================================================")
    pre_train(args, args.model2, pre_snapshot_path2)
    logging.info("====================================================================================")
    pre_train(args, args.model3, pre_snapshot_path3)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path1, pre_snapshot_path2, pre_snapshot_path3, self_snapshot_path)


if __name__ == "__main__":
    print('winwinwin')


    # labelnum = 3
    # for exp in range(5):
    #     run_main(exp, labelnum)
    # labelnum = 7
    # for exp in range(5):
    #     run_main(exp, labelnum)
    labelnum = 14
    for exp in range(4,5):
        run_main(exp, labelnum)







