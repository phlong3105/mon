import argparse
import csv
import itertools
import os
import test


if __name__ == '__main__':
    weights = [
        os.path.join("runs", "finetune", "yolov4-p7_a2i2haze_1536/weights/best.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_a2i2haze_1920/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_1536_a2i2haze_1536/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_1845_a2i2haze_1536/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_1845_a2i2haze_1845/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_1920_a2i2haze_1536/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2160_a2i2haze_1536/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2160_a2i2haze_1536_2/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2160_a2i2haze_1536_3/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2160_a2i2haze_1845/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2160_a2i2haze_multiscale/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_uavdt_1536_a2i2haze_1536/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2560_a2i2haze_1536/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2560_a2i2haze_1845/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2560_a2i2haze_1920/weights/best_strip.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2560_a2i2haze_2160/weights/best.pt"),
        os.path.join("runs", "finetune", "yolov4-p7_visdrone_2560_a2i2haze_2560/weights/best.pt"),
    ]
    
    N = 4  # Number of models in ensemble
    table = {}
    for comb in itertools.combinations(weights, n):
        parser = argparse.ArgumentParser(prog='ensemble.py')
        parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
        parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
        parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
        parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
        parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--merge', action='store_true', help='use Merge NMS')
        parser.add_argument('--verbose', action='store_true', help='report mAP by class')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        opt            = parser.parse_args()
        opt.save_json |= opt.data.endswith('coco.yaml')
            
        opt.data       = "data/a2i2haze.yaml"
        opt.weights    = list(comb)
        opt.batch_size = 1
        opt.img_size   = 1536
        opt.conf_thres = 0.001
        opt.iou_thres  = 0.5
        # opt.device     = "0"
        opt.augment    = True
        opt.merge      = True
        opt.verbose    = False
        print(opt)
            
        results = test.test(
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            opt.verbose,
            opt2=opt
        )
    
        k = ""
        for v in comb:
            v = v.replace("/weights/best.pt", "")
            v = v.replace("/weights/best_strip.pt", "")
            v = v.replace("runs/finetune/", "")
            k = f"{k} + {v}"
        table[k] = results
        
    # (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    header = ["ensemble", "mp", "mr", "map50", "map", "t"]
    with open(f"{N}C{len(weights)}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
            
        for k, v in table.items():
            v0    = v[0]
            mp    = v0[0]
            mr    = v0[1]
            map50 = v0[2]
            map   = v0[3]
            loss  = v0[4]
            maps  = v[1]
            t     = v[2]
            writer.writerow([
                f"{k}", f"{mp}", f"{mr}", f"{map50}", f"{map}", f"{t}"
            ])
