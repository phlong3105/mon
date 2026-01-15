import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )
            if "depth_pred" in sample:
                sample["depth_pred"] = cv2.resize(
                    sample["depth_pred"], (width, height), interpolation=cv2.INTER_NEAREST
                )
            if "semseg_mask" in sample:
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        if "depth_pred" in sample:
            sample["depth_pred"] = sample["depth_pred"].astype(np.float32)
            sample["depth_pred"] = np.ascontiguousarray(sample["depth_pred"])

        return sample


def crop_ori(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def crop(sample, size): 
    w, h = sample['image'].size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    sample['image'] = ImageOps.expand(sample['image'], border=(0, 0, padw, padh), fill=0)
    if 'image_enhanced' in sample.keys():
        sample['image_enhanced'] = ImageOps.expand(sample['image_enhanced'], border=(0, 0, padw, padh), fill=0)
    if 'depth' in sample.keys():
        sample['depth'] = ImageOps.expand(sample['depth'], border=(0, 0, padw, padh), fill=0)
  
    w, h = sample['image'].size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    sample['image'] = sample['image'].crop((x, y, x + size, y + size))
    if 'image_enhanced' in sample.keys():
        sample['image_enhanced'] = sample['image_enhanced'].crop((x, y, x + size, y + size))

    if 'depth' in sample.keys():
        sample['depth'] = sample['depth'].crop((x, y, x + size, y + size))
    return sample


def hflip_ori(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img, mask


def hflip(sample, p=0.5):
    if random.random() < p:
        sample['image'] = sample['image'].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if 'image_enhanced' in sample.keys():
            sample['image_enhanced'] = sample['image_enhanced'].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if 'depth' in sample.keys():
            sample['depth'] = sample['depth'].transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return sample


def normalize_ori(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def normalize(sample):
    sample['image'] = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(sample['image'])
    if 'image_enhanced' in sample.keys():
        sample['image_enhanced'] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(sample['image_enhanced'])
    if 'depth' in sample.keys():
        sample['depth'] = torch.from_numpy(np.array(sample['depth'])).float()
    return sample


def normalize_depth_only(sample,dataset=None):
    if dataset in ['sintel','DIODE','ETH3D']:
        sample['depth'] = torch.from_numpy(sample['depth']).float()
        
        return sample
    if sample['depth'] is not None:
        sample['depth'] = torch.from_numpy(np.array(sample['depth'])).float()
        
        return sample
    return sample


def resize_ori(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.Resampling.BILINEAR)
    mask = mask.resize((ow, oh), Image.Resampling.NEAREST)
    return img, mask


def resize(sample, ratio_range):
    img = sample['image']
    if 'depth' in sample.keys():
        mask = sample['depth']
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.Resampling.BILINEAR)
    if 'depth' in sample.keys():
        mask = mask.resize((ow, oh), Image.Resampling.NEAREST)
    sample['image'] = img
    if 'depth' in sample.keys():
        sample['depth'] = mask
    return sample


def resize_certain(sample, short_side):
    img = sample['image']
    if 'image_enhanced' in sample.keys():
        img_enhanced = sample['image_enhanced']
    if 'depth' in sample.keys():
        mask = sample['depth']
    
    w, h = img.size

    if h < w:
        oh = short_side
        ow = int(1.0 * w * short_side / h + 0.5)
    else:
        ow = short_side
        oh = int(1.0 * h * short_side / w + 0.5)

    

    assert isinstance(ow, int) and isinstance(oh, int), "ow and oh must be integers"

    img = img.resize((ow, oh), Image.Resampling.BILINEAR)

    if 'image_enhanced' in sample.keys():
        img_enhanced = img_enhanced.resize((ow, oh), Image.Resampling.BILINEAR)
        sample['image_enhanced'] = img_enhanced
        
    if 'depth' in sample.keys():
        
        mask = mask.resize((ow, oh), Image.Resampling.NEAREST)
    sample['image'] = img
    if 'depth' in sample.keys():
        sample['depth'] = mask
    return sample


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=int(sigma)))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)


def img_aug_identity(img, scale=None):
    return img


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.05, 0.95], p=0.2):
    if random.random() < p:
        min_v, max_v = min(scale), max(scale)
        v = float(max_v - min_v) * random.random()
        v = max_v - v
        return ImageEnhance.Contrast(img).enhance(v)
    else:
        return img


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    return ImageOps.solarize(img, v)


def get_augment_list():
    l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.05, 0.95]),
        (img_aug_brightness, [0.05, 0.95]),
        (img_aug_color, [0.05, 0.95]),
        (img_aug_sharpness, [0.05, 0.95]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5])
    ]
    return l


class strong_img_aug:
    
    def __init__(self, num_augs=4, flag_using_random_num=True):
        self.n = num_augs
        self.augment_list = get_augment_list()
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img
