import argparse

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.data import *
from loss.losses import *
from net.cidnet import CIDNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

eval_parser = argparse.ArgumentParser(description="Eval")
eval_parser.add_argument("--perc",              action="store_true", help="trained with perceptual loss")
eval_parser.add_argument("--lol_v1",            action="store_true", help="output lolv1 dataset")
eval_parser.add_argument("--lol_v2_real",       action="store_true", help="output lol_v2_real dataset")
eval_parser.add_argument("--lol_v2_synthetic",  action="store_true", help="output lol_v2_synthetic dataset")
eval_parser.add_argument("--sice_grad",         action="store_true", help="output sice_grad dataset")
eval_parser.add_argument("--sice_mix",          action="store_true", help="output sice_mix dataset")

eval_parser.add_argument("--best_gt_mean",      action="store_true", help="output lol_v2_real dataset best_gt_mean")
eval_parser.add_argument("--best_psnr",         action="store_true", help="output lol_v2_real dataset best_psnr")
eval_parser.add_argument("--best_ssim",         action="store_true", help="output lol_v2_real dataset best_ssim")

eval_parser.add_argument("--unpaired",          action="store_true", help="output unpaired dataset")
eval_parser.add_argument("--dicm",              action="store_true", help="output DICM dataset")
eval_parser.add_argument("--lime",              action="store_true", help="output LIME dataset")
eval_parser.add_argument("--mef",               action="store_true", help="output MEF dataset")
eval_parser.add_argument("--npe",               action="store_true", help="output NPE dataset")
eval_parser.add_argument("--vv",                action="store_true", help="output VV dataset")
eval_parser.add_argument("--alpha",             type=float, default=1.0)
eval_parser.add_argument("--unpaired_weights",  type=str,   default="./weights/lol_v2_synthetic/w_perc.pth")

ep = eval_parser.parse_args()


def eval(
    model,
    testing_data_loader,
    model_path,
    output_folder,
    norm_size = True,
    lol_v1    = False,
    lol_v2    = False,
    unpaired  = False,
    alpha     = 1.0
):
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    # print("Pre-trained model is loaded.")
    torch.set_grad_enabled(False)
    model.eval()
    # print("Evaluation:")
    if lol_v1:
        model.trans.gated  = True
    elif lol_v2:
        model.trans.gated2 = True
        model.trans.alpha  = alpha
    elif unpaired:
        model.trans.alpha  = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input  = input.cuda()
            output = model(input) 
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(), 0, 1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    # print("===> End evaluation")
    if lol_v1:
        model.trans.gated  = False
    elif lol_v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)


if __name__ == "__main__":
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists("./output"):          
        os.mkdir("./output")  
    
    norm_size   = True
    num_workers = 1
    alpha       = None
    if ep.lol_v1:
        eval_data     = DataLoader(dataset=get_eval_set("../datasets/LOLdataset/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = "./output/lol_v1/"
        if ep.perc:
            weight_path = "./weights/lol_v1/w_perc.pth"
        else:
            weight_path = "./weights/lol_v1/wo_perc.pth"
        weight_path = "./weights/train/epoch_500.pth"
    elif ep.lol_v2_real:
        eval_data     = DataLoader(dataset=get_eval_set("../datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = "./output/LOLv2_real/"
        if ep.best_gt_mean:
            weight_path = "./weights/LOLv2_real/w_perc.pth"
            alpha       = 0.84
        elif ep.best_psnr:
            weight_path = "./weights/LOLv2_real/best_psnr.pth"
            alpha       = 0.8
        elif ep.best_ssim:
            weight_path = "./weights/LOLv2_real/best_ssim.pth"
            alpha       = 0.82
    elif ep.lol_v2_synthetic:
        eval_data = DataLoader(dataset=get_eval_set("../datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = "./output/LOLv2_syn/"
        if ep.perc:
            weight_path = "./weights/LOLv2_syn/w_perc.pth"
        else:
            weight_path = "./weights/LOLv2_syn/DVCNet_epoch_320_best.pth"
    elif ep.sice_grad:
        eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/SICE/SICE_Grad"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = "./output/sice_grad/"
        weight_path   = "./weights/SICE.pth"
        norm_size     = False
    elif ep.sice_mix:
        eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/SICE/SICE_Mix"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = "./output/sice_mix/"
        weight_path   = "./weights/SICE.pth"
        norm_size     = False
    elif ep.unpaired: 
        if ep.dicm:
            eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/DICM"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = "./output/DICM/"
        elif ep.lime:
            eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/LIME"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = "./output/LIME/"
        elif ep.mef:
            eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/MEF"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = "./output/MEF/"
        elif ep.npe:
            eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/NPE"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = "./output/NPE/"
        elif ep.vv:
            eval_data     = DataLoader(dataset=get_sice_eval_set("../datasets/VV"), num_workers=num_workers, batch_size=1, shuffle=False)
            output_folder = "./output/VV/"
        alpha       = ep.alpha
        norm_size   = False
        weight_path = ep.unpaired_weights
        
    eval_net = CIDNet().cuda()
    eval(
        eval_net,
        eval_data,
        weight_path,
        output_folder,
        norm_size = norm_size,
        lol_v1    = ep.lol_v1,
        lol_v2    = ep.lol_v2_real,
        unpaired  = ep.unpaired,
        alpha     = alpha
    )
