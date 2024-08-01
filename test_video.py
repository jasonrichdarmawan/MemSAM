import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.model_dict import get_model
from utils.config import get_config
from utils.data_us import EchoDataset, EchoVideoDataset, JointTransform3D
from utils.evaluation import get_eval
from utils.loss_functions.sam_loss import get_criterion


def parse_args():
    parser = argparse.ArgumentParser(description="Networks")
    parser.add_argument(
        "--modelname",
        default="MemSAM",
        type=str,
        help="type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS, MemSAM...",
    )
    parser.add_argument(
        "--encoder_input_size",
        type=int,
        default=256,
        help="the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS",
    )
    parser.add_argument(
        "--low_image_size",
        type=int,
        default=256,
        help="the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS",
    )
    parser.add_argument(
        "--task",
        default="CAMUS_Video_Semi",
        help="task: CAMUS_Video_Full, CAMUS_Video_Semi or EchoNet_Video",
    )
    parser.add_argument(
        "--sam_ckpt",
        type=str,
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="Pretrained checkpoint of SAM",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/CAMUS_full/your_checkpoint.pth",
        help="",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size per gpu")
    parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
    parser.add_argument("--compute_ef", action="store_true")
    parser.add_argument("--disable_memory", action="store_true")
    parser.add_argument("--disable_reinforce", action="store_true")
    parser.add_argument("--disable_point_prompt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    opt = get_config(args.task)  # please configure your hyper-parameter

    print("task", args.task, "checkpoints:", args.ckpt_path)
    opt.mode = "test"
    device = torch.device(opt.device)

    # ==================================================set random seed==================================================
    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    opt.batch_size = args.batch_size * args.n_gpu

    tf_val = JointTransform3D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True,
    )
    test_dataset = EchoVideoDataset(
        opt.data_path,
        opt.test_split,
        tf_val,
        img_size=args.encoder_input_size,
        frame_length=opt.frame_length,
        point_numbers=opt.point_prompt_number,
        disable_point_prompt=args.disable_point_prompt,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = get_model(args.modelname, args=args)
    model.to(device)
    model.train()

    checkpoint = torch.load(args.ckpt_path)
    # ------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    #  ========================================================================= begin to evaluate the model ============================================================================

    #   ======================== Model trainable parameters ==============================
    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total_params: {}".format(pytorch_total_params))

    #   ======================== test Model Gflops / GPU Mem Usage ==============================
    # input = torch.randn(1, 1, 3, args.encoder_input_size, args.encoder_input_size).cuda()
    # points = (torch.tensor([[[[1, 2]]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    # from thop import profile
    # flops, params = profile(model, inputs=(input, points), )
    # print('Gflops:', flops/1000000000, 'params:', params)

    #   ======================== test inference time ==============================
    # sum_time = 0
    # with torch.no_grad():
    #     start_time = time.time()
    #     pred = model(input, points, None)
    #     sum_time =  sum_time + (time.time()-start_time)
    # print("test speed", sum_time)

    model.eval()
    dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std = (
        get_eval(testloader, model, criterion=criterion, opt=opt, args=args)
    )
    print("dataset:" + args.task + " -----------model name: " + args.modelname)
    print("task", args.task, "checkpoints:", args.ckpt_path)
    print("dice_mean  iou_mean  hd_mean  assd_mean")
    print(dice_mean, iou_mean, hd_mean, assd_mean)
    print("dices_std  iou_std  hd_std  assd_std")
    print(dices_std, iou_std, hd_std, assd_std)


if __name__ == "__main__":
    main()
