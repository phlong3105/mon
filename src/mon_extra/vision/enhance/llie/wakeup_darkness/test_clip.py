import os
import sys
import numpy as np
import torch
import argparse
from PIL import Image, ImageChops, ImageDraw
from torch.autograd import Variable
from model_cvpr import Network_woCalibrate
from dataset_eccv import ImageLowSemDataset,ImageLowSemDataset_Val
import cv2
from segment_anything import build_sam, SamAutomaticMaskGenerator
from typing import List
import clip

# 定义全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clip, preprocess = clip.load("ViT-B/16", device=device)

parser = argparse.ArgumentParser("enlighten-anything")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--weights', type=str, default="./exp/SICE/Train-20240126-143807/model_epochs/weights_0.pt", help='weights after training with semantic')
#parser.add_argument('--weights', type=str, default="./weights/pretrained_SCI/easy.pt", help='weights after training with semantic')

parser.add_argument('--test_dir', type=str, default='./data/LOL/test15/low', help='testing data directory')
parser.add_argument('--test_output_dir', type=str, default='./ECCV/LOL_clip', help='testing output directory')
args = parser.parse_args()

save_path = args.test_output_dir
os.makedirs(save_path, exist_ok=True)

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def model_init(model):
    weights_dict = torch.load(args.weights)
    model_dict = model.state_dict()
    weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict)

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

@torch.no_grad()
def retriev(elements: List[Image.Image], search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model_clip.encode_image(stacked_images)
    text_features = model_clip.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    TestDataset = ImageLowSemDataset_Val(img_dir=args.test_dir, sem_dir = os.path.join(os.path.split(args.test_dir)[0], 'low_semantic'), depth_dir= os.path.join(os.path.split(args.test_dir)[0], 'low_depth'))
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1, shuffle=False,
        pin_memory=True
    )


    model = Network_woCalibrate()
    model_init(model)
    model = model.cuda()

    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_clip, preprocess = clip.load("ViT-B/16", device=device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (in_, sem_, imgname_, semname_ ) in enumerate(test_queue):
            in_ = in_.cuda()
            sem_ = sem_.cuda()
            image_name = os.path.splitext(imgname_[0])[0]
            i, r = model(in_, sem_)
            u_name = '%s.png' % (image_name)
            print('test processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)

            image_path = os.path.join(save_path, u_name)
            original_image_path = image_path
            new_image_path = os.path.join(args.test_dir, os.path.basename(imgname_[0]))

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)

            image = Image.open(image_path)
            cropped_boxes = []

            for mask in masks:
                cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))

            scores = retriev(cropped_boxes, "a photo of a things")
            indices = get_indices_of_values_above_threshold(scores, 0.05)

            segmentation_masks = []

            for seg_idx in indices:
                segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
                segmentation_masks.append(segmentation_mask_image)

            original_image = Image.open(image_path)
            overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_color = (255, 0, 0, 200)

            draw = ImageDraw.Draw(overlay_image)
            for segmentation_mask_image in segmentation_masks:
                draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

            original_image = Image.open(original_image_path).convert('RGBA')
            new_image = Image.open(new_image_path).convert('RGBA')

            combined_mask = Image.new('L', original_image.size, 0)

            for segmentation_mask_image in segmentation_masks:
                combined_mask = ImageChops.lighter(combined_mask, segmentation_mask_image.convert('L'))

            extracted_region = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
            extracted_region.paste(original_image, mask=combined_mask)

            new_image = Image.alpha_composite(new_image, extracted_region)

            result_image = new_image.convert('RGB')

            result_image_path = os.path.join(save_path, 'output_{}.jpg'.format(batch_idx))
            result_image.save(result_image_path)

if __name__ == '__main__':
    main()

