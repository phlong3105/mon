import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from onnx2tflite.converter import onnx_converter
onnx_path = "/data2/yanhailong/IR-Based/ICCV2025/MobileIE/LLE.onnx" 

onnx_converter(
    onnx_model_path = onnx_path,
    need_simplify = True,
    output_path = "//data2/yanhailong/IR-Based/ICCV2025/MobileIE/", 
    target_formats = ['tflite'], # or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = False,
    int8_mean = None,
    int8_std = None,
    image_root = None
)