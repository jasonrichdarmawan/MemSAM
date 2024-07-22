class Config_CAMUS_Video_Full():
    data_path = "/data/dengxiaolong/memsam/CAMUS_public/" # CAMUS dataset path
    save_path = "./checkpoints/CAMUS_Video_Full/"
    result_path = "./result/CAMUS_Video_Full/"
    tensorboard_path = "./tensorboard/CAMUS_Video_Full/"

    img_size = 256
    frame_length = 10                       # video sample frame length
    loss_pred_idx = [i for i in range(10)]  # use pred-frame idx for loss calc
    loss_label_idx = [i for i in range(10)] # use label-frame idx for loss calc
    pred_idx = [i for i in range(10)]       # use pred-frame idx for val
    label_idx = [i for i in range(10)]      # use all label-frame idx for val
    crop = None                             # the cropped image size
    train_split = "train"                   # the file name of training set
    val_split = "val"                       # the file name of testing set
    test_split = "test"                     # the file name of testing set
    classes = 2                             # thenumber of classes (background + foreground)
    device = "cuda"                         # training device, cpu or cuda
    point_prompt_number = 1                 # use gt point prompt number
    eval_freq = 1
    save_freq = 2000
    eval_mode = "camus"
    pre_trained = None


class Config_CAMUS_Video_Semi():
    data_path = "/data/dengxiaolong/memsam/CAMUS_public/" # CAMUS dataset path
    save_path = "./checkpoints/CAMUS_Video_Semi/"
    result_path = "./result/CAMUS_Video_Semi/"
    tensorboard_path = "./tensorboard/CAMUS_Video_Semi/"

    img_size = 256
    frame_length = 10                       # video sample frame length
    loss_pred_idx = [0,9]                   # use pred-frame idx for loss calc
    loss_label_idx = [0,9]                  # use label-frame idx for loss calc
    pred_idx = [i for i in range(10)]       # use pred-frame idx for val
    label_idx = [i for i in range(10)]      # use all label-frame idx for val
    crop = None                             # the cropped image size
    train_split = "train"                   # the file name of training set
    val_split = "val"                       # the file name of testing set
    test_split = "test"                     # the file name of testing set
    classes = 2                             # thenumber of classes (background + foreground)
    device = "cuda"                         # training device, cpu or cuda
    point_prompt_number = 1                 # use gt point prompt number
    eval_freq = 1
    save_freq = 2000
    eval_mode = "camus"
    pre_trained = None


class Config_EchoNet_Video():
    data_path = "/data/dengxiaolong/memsam/EchoNet/echocycle/"  # EchoNet-Dynamic dataset path
    save_path = "./checkpoints/EchoNet_Video/"
    result_path = "./result/EchoNet_Video/"
    tensorboard_path = "./tensorboard/EchoNet_Video/"

    img_size = 128
    frame_length = 10                       # video sample frame length
    loss_pred_idx = [0,9]                   # use pred-frame idx for loss calc
    loss_label_idx = [0,1]                  # use label-frame idx for loss calc
    pred_idx = [0,9]                        # use pred-frame idx for val
    label_idx = [0,1]                       # use all label-frame idx for val
    crop = None                             # the cropped image size
    train_split = "train"                   # the file name of training set
    val_split = "val"                       # the file name of testing set
    test_split = "test"                     # the file name of testing set
    classes = 2                             # thenumber of classes (background + foreground)
    device = "cuda"                         # training device, cpu or cuda
    point_prompt_number = 1                 # use gt point prompt number
    eval_freq = 1
    save_freq = 2000
    eval_mode = "echonet"
    pre_trained = None


# ==================================================================================================
def get_config(task="CAMUS_Video_Full"):
    if task == "EchoNet_Video":
        return Config_EchoNet_Video()
    elif task == "CAMUS_Video_Full":
        return Config_CAMUS_Video_Full()
    elif task == "CAMUS_Video_Semi":
        return Config_CAMUS_Video_Semi()
    else:
        assert("We do not have the related dataset, please choose another task.")