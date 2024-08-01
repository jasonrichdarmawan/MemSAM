import os

import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

np.seterr(divide='ignore', invalid='ignore')
import time

import torch
import torch.nn.functional as F
from einops import rearrange
from hausdorff import hausdorff_distance
from medpy.metric.binary import assd as medpy_assd
from medpy.metric.binary import hd as medpy_hd
from medpy.metric.binary import hd95 as medpy_hd95
from scipy import stats

import utils.metrics as metrics
from utils.compute_ef import compute_left_ventricle_volumes
from utils.generate_prompts import get_click_prompt
from utils.tools import bias, corr
from utils.tools import hausdorff_distance as our_hausdorff_distance
from utils.tools import std
from utils.visualization import (visual_segmentation,
                                 visual_segmentation_binary,
                                 visual_segmentation_npy,
                                 visual_segmentation_sets,
                                 visual_segmentation_sets_with_pt)


def eval_camus(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0.0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        # video to image
        b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        patient_name = image_filename[0].split('.')[0].split('_')[0]
        view = image_filename[0].split('.')[0].split('_')[1]
        gt_efs[patient_name] = datapack['ef'].detach().cpu().numpy()[0]
        if args.disable_point_prompt:
            # pt[0]: b t 1 2
            # pt[1]: t 1
            pt = None
        else:
            pt = get_click_prompt(datapack, opt)

        # start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
        # end = time.time()
        # print('infer_time:', (end-start))
        # sum_time = sum_time + (end-start)

        val_loss = criterion(pred[:,opt.pred_idx,0], masks[:,opt.label_idx])
        pred = pred[:,opt.pred_idx, 0]
        masks = masks[:,opt.label_idx]
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred)
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if patient_name not in mask_dict:
            mask_dict[patient_name] = {}
        mask_dict[patient_name][view] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape

        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)
                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                if opt.mode == "test":
                    try:
                        med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                        med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    except:
                        print(pred_i[0], gt_i[0])
                        raise RuntimeError
                    hds.append(med_hd95)
                    assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                # dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                # print(dice)
                # if opt.visual:
                #     visual_segmentation_npy(pred_i[0,...], gt_i[0,...], image_filename[j], opt, imgs[j:j+1, frame_i, :, :, :], frameidx=frame_i)
    
    print('average_fps:', 1/ (sum_time / len(valloader) / 10) )
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    hds = np.array(hds)
    assds = np.array(assds)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
    elif opt.mode == "test":
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)
        if args.compute_ef:
            # compute ef
            pred_efs = {}
            for patient_name in mask_dict:
                a2c_ed = mask_dict[patient_name]['2CH']['ED']
                a2c_es = mask_dict[patient_name]['2CH']['ES']
                a2c_voxelspacing = mask_dict[patient_name]['2CH']['spacing']
                a4c_ed = mask_dict[patient_name]['4CH']['ED']
                a4c_es = mask_dict[patient_name]['4CH']['ES']
                a4c_voxelspacing = mask_dict[patient_name]['4CH']['spacing']
                edv, esv = compute_left_ventricle_volumes(
                    a2c_ed=a2c_ed,
                    a2c_es=a2c_es,
                    a2c_voxelspacing=a2c_voxelspacing,
                    a4c_ed=a4c_ed,
                    a4c_es=a4c_es,
                    a4c_voxelspacing=a4c_voxelspacing,
                )
                if esv > edv:
                    edv, esv = esv, edv
                ef = round(100 * (edv - esv) / edv, 2)
                pred_efs[patient_name] = ef
                print(patient_name, pred_efs[patient_name], gt_efs[patient_name])

            gt_ef_array = list(gt_efs.values())
            pred_ef_array = list(pred_efs.values())
            # gt_ef_array = [round(i) for i in gt_ef_array]
            # pred_ef_array = [round(i) for i in pred_ef_array]
            gt_ef_array = np.array(gt_ef_array)
            pred_ef_array = np.array(pred_ef_array)
            print(
                'bias:', bias(gt_ef_array,pred_ef_array),
                'std:', std(pred_ef_array),
                'corr', corr(gt_ef_array,pred_ef_array)
            )
            wilcoxon_rank_sum_test = stats.mannwhitneyu(gt_ef_array ,pred_ef_array)
            wilcoxon_signed_rank_test = stats.wilcoxon(gt_ef_array ,pred_ef_array)
            print(wilcoxon_rank_sum_test)
            print(wilcoxon_signed_rank_test)
        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std


def eval_echonet(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = np.zeros(opt.classes)
    tps, fps, tns, fns, hds, assds= [],[],[],[],[],[]
    mask_dict = {}
    gt_efs = {}
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
        spcaing = datapack['spacing'].detach().cpu().numpy()[0,:2][::-1] # remove z and reverse (y,x)
        # video to image
        # b, t, c, h, w = imgs.shape

        image_filename = datapack['image_name']
        image_name = image_filename[0].split(".")[0]

        gt_efs[image_name] = datapack['ef'].detach().cpu().numpy()[0]
        # if args.enable_point_prompt:
        #     # pt[0]: b t 1 2
        #     # pt[1]: t 1
        pt = get_click_prompt(datapack, opt)
        # else:
        # pt = None

        # start = time.time()
        with torch.no_grad():
            pred = model(imgs, pt, None)
        # end = time.time()
        # sum_time = sum_time +(end-start)
        # print('infer_time:', end-start)

        pred = pred[:, opt.pred_idx,0]
        masks = masks[:,opt.label_idx]

        gt = masks.detach().cpu().numpy()
        predict = F.sigmoid(pred)
        predict = predict.detach().cpu().numpy()  # (b, t, h, w)
        seg = predict > 0.6

        seg_mask = np.zeros_like(gt)
        seg_mask[seg] = 1
        if image_name not in mask_dict:
            mask_dict[image_name] = {}
        mask_dict[image_name] = {'ED':seg_mask[0,0], 'ES':seg_mask[0,-1],'spacing':spcaing}

        b, t, h, w = seg.shape
        for j in range(0, b):
            for idx, frame_i in enumerate(range(0,t)):
                # for idx, frame_i in enumerate([0,t-1]):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, frame_i,:, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, frame_i, :, :] == 1] = 255
                tp, fp, tn, fn = metrics.get_matrix(pred_i, gt_i)

                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                print(dice)
                # if opt.visual:
                #     visual_segmentation_npy(pred_i[0, ...],
                #                             gt_i[0, ...],
                #                             image_filename[j],
                #                             opt,
                #                             imgs[j:j + 1, frame_i, :, :, :],
                #                             frameidx=frame_i)
                # continue

                # hausdorff_distance
                # hd = hausdorff_distance(pred_i[0], gt_i[0], distance="euclidean")
                # our
                # our_hd = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=100)
                # our_hd_95 = our_hausdorff_distance(pred_i[0], gt_i[0], percentile=95)
                # medpy
                # med_hd = medpy_hd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                try:
                    med_hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spcaing)
                    med_assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spcaing)
                except:
                    print(pred_i[0], gt_i[0])
                    raise RuntimeError
                # print(med_hd95)
                # print(med_assd)
                hds.append(med_hd95)
                assds.append(med_assd)
                tps.append(tp)
                fps.append(fp)
                tns.append(tn)
                fns.append(fn)
                dice = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
                print(dice)

    print(sum_time / len(valloader))
    tps = np.array(tps)
    fps = np.array(fps)
    tns = np.array(tns)
    fns = np.array(fns)
    patient_dices = (2 * tps + 1e-5) / (2 * tps + fps + fns + 1e-5)  # p c
    #return dices, mean_dice, val_losses
    if opt.mode == "train":
        dices = np.mean(patient_dices, axis=0)  # c
        hdis = np.mean(hds, axis=0)
        val_losses = val_losses / (batch_idx + 1)
        mean_dice = dices[0]
        mean_hdis = hdis
        return dices, mean_dice, mean_hdis, val_losses
    else:
        # hds = np.array(hds)
        # assds = np.array(assds)
        dice_mean = np.mean(patient_dices, axis=0)
        dices_std = np.std(patient_dices, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        smooth = 0.00001
        iou = (tps + smooth) / (fps + tps + fns + smooth) # p c
        iou_mean = np.mean(iou, axis=0)
        iou_std = np.std(iou, axis=0)
        assd_mean = np.mean(assds, axis=0)
        assd_std = np.std(assds, axis=0)
        if args.compute_ef:
            # compute ef
            pred_efs = {}
            for patient_name in mask_dict:
                a2c_ed = a4c_ed = mask_dict[patient_name]['ED']
                a2c_es = a4c_es = mask_dict[patient_name]['ES']
                a2c_voxelspacing = a4c_voxelspacing = mask_dict[patient_name]['spacing']
                edv, esv = compute_left_ventricle_volumes(
                    a2c_ed=a2c_ed,
                    a2c_es=a2c_es,
                    a2c_voxelspacing=a2c_voxelspacing,
                    a4c_ed=a4c_ed,
                    a4c_es=a4c_es,
                    a4c_voxelspacing=a4c_voxelspacing,
                )
                ef = round(100 * (edv - esv) / edv, 2)
                pred_efs[patient_name] = ef
                print(patient_name, pred_efs[patient_name], gt_efs[patient_name])

            gt_ef_array = list(gt_efs.values())
            pred_ef_array = list(pred_efs.values())
            # gt_ef_array = [round(i) for i in gt_ef_array]
            # pred_ef_array = [round(i) for i in pred_ef_array]
            gt_ef_array = np.array(gt_ef_array)
            pred_ef_array = np.array(pred_ef_array)
            print(
                'bias:', bias(gt_ef_array,pred_ef_array),
                'std:', std(pred_ef_array),
                'corr', corr(gt_ef_array,pred_ef_array)
            )
            wilcoxon_rank_sum_test = stats.mannwhitneyu(gt_ef_array ,pred_ef_array)
            wilcoxon_signed_rank_test = stats.wilcoxon(gt_ef_array ,pred_ef_array)
            print(wilcoxon_rank_sum_test)
            print(wilcoxon_signed_rank_test)
        return dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std


def get_eval(val_loader, model, criterion, opt, args):
    if opt.eval_mode == "echonet":
        return eval_echonet(val_loader, model, criterion, opt, args)
    elif opt.eval_mode == "camus":
        return eval_camus(val_loader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)