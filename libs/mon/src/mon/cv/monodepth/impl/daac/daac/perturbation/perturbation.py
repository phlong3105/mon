import random

import numpy as np
import yaml
from imagecorruptions import corrupt
from PIL import Image
from torchvision import transforms


def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a+1e-5)) ** gamma) * (d - c) + c
    return y


def poisson_gaussian_noise(path,x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x)
    if x.max() > 255 or x.min() < 0:
        print(f"Warning: {path} Input image values out of range: min={x.min()}, max={x.max()}")

    x = np.clip(x, 0, 255) / 255.
    try:
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
        c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    except Exception as e:
        print("Error image path:", path)
        print("Poisson noise parameter c_poisson:", c_poisson)
        print("Noise level severity:", severity)
        print("Image data x:", x)
        print("Image shape x.shape:", x.shape)
        print("Image minimum value x.min():", x.min())
        print("Image maximum value x.max():", x.max())
        print("Gaussian noise parameter c_gauss:", c_gauss)
        print(f"Error occurred during noise addition: {str(e)}")
    return Image.fromarray(np.uint8(x))


def low_light(path, x, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity-1]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2) * 255
    x_scaled = poisson_gaussian_noise(path,x_scaled, severity=severity-1)
    return x_scaled


def create_dark(path, image, severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Dark'.
    Parameters:
        image: Input image
        severity_lower: Lower bound of severity (inclusive)
        severity_upper: Upper bound of severity (inclusive)
    Returns:
        Image with dark lighting interference
    """
    severity = random.randint(severity_lower, severity_upper)
    corrupted = low_light(path,image, severity=severity)
    return corrupted


def create_fog(path, image, severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Fog'.
    """
    corruption = 'fog'
    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    try:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
    except Exception as e:
        print("Error image path:", path)
        print("Fog parameter severity:", severity)
        print(f"Error occurred during noise addition: {str(e)}")
    im = transforms.ToPILImage()(corrupted)       
    return im


def create_frost(path,image,severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Frost'.
    """
    corruption = 'frost'
    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    try:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
        im = transforms.ToPILImage()(corrupted)
    except Exception as e:
        print("Error image path:", path)
        print("Frost parameter severity:", severity)
        print(f"Error occurred during noise addition: {str(e)}")    
    return im


def create_snow(path,image,severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Snow'.
    """
    corruption = 'snow'
    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    try:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
        im = transforms.ToPILImage()(corrupted)
    except Exception as e:
        print("Error image path:", path)
        print("Snow parameter severity:", severity)
        print(f"Error occurred during noise addition: {str(e)}")
    return im


def create_motion_blur(path,image,severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Motion Blur'.
    """
    corruption = 'motion_blur'
    severity = random.randint(severity_lower, severity_upper)
    image = np.array(image)
    try:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
        im = transforms.ToPILImage()(corrupted)
    except Exception as e:
        print("Error image path:", path)
        print("Motion blur parameter severity:", severity)
        print(f"Error occurred during noise addition: {str(e)}")
    return im


def create_zoom_blur(path,image,severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Zoom Blur'.
    """
    corruption = 'zoom_blur'
    severity = random.randint(severity_lower, severity_upper)
    image=np.array(image)
    try:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
        im = transforms.ToPILImage()(corrupted)
    except Exception as e:
        print("Error image path:", path)
        print("Zoom blur parameter severity:", severity)
        print(f"Error occurred during noise addition: {str(e)}")  
    im = transforms.ToPILImage()(corrupted)
    return im


def create_contrast(path,image,severity_lower=1, severity_upper=5):
    """
    Create corruptions: 'Contrast'.
    """
    corruption = 'contrast'
    severity = random.randint(severity_lower, severity_upper)
    image=np.array(image)
    try:
        corrupted = corrupt(image, corruption_name=corruption, severity=severity)
        im = transforms.ToPILImage()(corrupted)
    except Exception as e:
        print("Error image path:", path)
        print("Contrast parameter severity:", severity)
        print(f"Error occurred during noise addition: {str(e)}")
    im = transforms.ToPILImage()(corrupted)
    return im


def perturbation(path, image, cfg):
    if cfg['no_argu']:
        return image
    if cfg['is_dark']:
        corrupted = create_dark(path, image, severity_lower=cfg['severity_lower_dark'], severity_upper=cfg['severity_upper_dark'])
    if cfg['is_contrast']:
        if random.random() < cfg['prob_contrast']:
            corrupted = create_contrast(path, corrupted, severity_lower=cfg['severity_lower_contrast'], severity_upper=cfg['severity_upper_contrast'])

    weather_effects = []
    if cfg['is_fog']:
        weather_effects.append(('fog', cfg['prob_fog']))
    if cfg['is_frost']:  
        weather_effects.append(('frost', cfg['prob_frost']))
    if cfg['is_snow']:
        weather_effects.append(('snow', cfg['prob_snow']))
        
    if weather_effects:
        effect, prob = random.choice(weather_effects)
        if random.random() < prob:
            if effect == 'fog':
                corrupted = create_fog(path, corrupted, severity_lower=cfg['severity_lower_fog'], severity_upper=cfg['severity_upper_fog'])
            elif effect == 'frost':
                corrupted = create_frost(path, corrupted, severity_lower=cfg['severity_lower_frost'], severity_upper=cfg['severity_upper_frost'])
            elif effect == 'snow':
                corrupted = create_snow(path, corrupted, severity_lower=cfg['severity_lower_snow'], severity_upper=cfg['severity_upper_snow'])

    if cfg['is_motion_blur'] and cfg['is_zoom_blur']:
        if random.random() < 0.5:
            if random.random() < cfg['prob_motion_blur']:
                corrupted = create_motion_blur(path, corrupted, severity_lower=cfg['severity_lower_motion_blur'], severity_upper=cfg['severity_upper_motion_blur'])
        else:
            if random.random() < cfg['prob_zoom_blur']:
                corrupted = create_zoom_blur(path, corrupted, severity_lower=cfg['severity_lower_zoom_blur'], severity_upper=cfg['severity_upper_zoom_blur'])
    else:
        if cfg['is_motion_blur'] and random.random() < cfg['prob_motion_blur']:
            corrupted = create_motion_blur(path, corrupted, severity_lower=cfg['severity_lower_motion_blur'], severity_upper=cfg['severity_upper_motion_blur'])
        elif cfg['is_zoom_blur'] and random.random() < cfg['prob_zoom_blur']:
            corrupted = create_zoom_blur(path, corrupted, severity_lower=cfg['severity_lower_zoom_blur'], severity_upper=cfg['severity_upper_zoom_blur'])
    return corrupted


if __name__ == "__main__":
    cfg   = yaml.load(open("configs/depthanything_AC_vits.yaml", "r"), Loader=yaml.FullLoader)
    image = Image.open("your/path/to/image.png")
    corrupted_dark = create_dark("your/path/to/image.png",image,severity_lower=cfg['severity_lower_dark'], severity_upper=cfg['severity_upper_dark'])
    corrupted_dark.save("your/path/to/image_save.png")
