import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
from .dataloader import RandomLoader, ImageLoader

from onnx2tflite.utils import OPERATOR
from onnx2tflite.layers import conv_layers
from onnx2tflite.utils.definitions import *
from onnx2tflite.utils.graph_tools import build_tf_inputs, decode_node_attribute

import logging

# 设置日志
LOG = logging.getLogger("keras_builder")
LOG.setLevel(logging.INFO)  # 设定日志级别

# 添加日志处理器（如果未添加）
if not LOG.hasHandlers():
    handler = logging.StreamHandler()  # 输出到控制台
    formatter = logging.Formatter("%(levelname)s: %(message)s")  # 设置日志格式
    handler.setFormatter(formatter)
    LOG.addHandler(handler)


def keras_builder(onnx_model, native_groupconv:bool=False):
    """
    将 ONNX 模型转换为 Keras 模型。
    
    参数:
    - onnx_model (onnx.ModelProto): 需要转换的 ONNX 模型。
    - native_groupconv (bool, 可选): 是否保持 ONNX 原生的分组卷积操作。默认 False。

    返回:
    - keras_model (keras.Model): 生成的 Keras 模型。
    - input_layout (dict): ONNX 输入张量的布局信息。
    - output_layout (dict): ONNX 输出张量的布局信息。
    """
    # 设置全局变量，控制是否使用原生 ONNX 分组卷积
    conv_layers.USE_NATIVE_GROUP_CONV = native_groupconv

    # 解析 ONNX 计算图
    model_graph = onnx_model.graph
    layout_dict, tf_tensor = {}, {} # 存储 ONNX 层的布局信息 & TensorFlow 层的映射
    
    '''
        初始化 ONNX 的权重张量
    '''

    onnx_weights = dict()
    
    for initializer in model_graph.initializer:
        # 将 ONNX 权重转换为 NumPy 数组
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)

    '''
        解析 ONNX 输入节点并转换为 TensorFlow 输入层
    '''
    input_nodes = build_tf_inputs(model_graph, layout_dict) # 解析 ONNX 输入
    tf_tensor.update(input_nodes)                           # 更新 TensorFlow 层的映射字典

    '''
        遍历 ONNX 计算图中的所有节点，并转换为 TensorFlow 层
    '''

    ###########################################################################################################
    for node in model_graph.node:

        op_name, node_inputs, node_outputs = node.op_type, node.input, node.output
        op_attr = decode_node_attribute(node) # 解析 ONNX 节点的属性
        
        # 查找 TensorFlow 对应的操作
        tf_operator = OPERATOR.get(op_name)
        if tf_operator is None:
            raise KeyError(f"{op_name} not implemented yet")
        
        _inputs = None 
        if len(node_inputs) > 0:  # 如果输入张量已在 `tf_tensor` 中，使用它，否则从 `onnx_weights` 取出
            _inputs = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else onnx_weights[node_inputs[0]]

        # 初始化 layout（数据格式，例如 NHWC）
        for index in range(len(node_outputs)):
            layout_dict[node_outputs[index]] = layout_dict.get(node_inputs[0], Layout.Default)

        # 执行转换：ONNX 层 -> TensorFlow 层
        res = tf_operator(tf_tensor, onnx_weights, node_inputs, op_attr, node_outputs, layout_dict)(_inputs)
        
    ###########################################################################################################
        if isinstance(res, list): # 处理多个输出
            for index in range(len(node_outputs)):
                tf_tensor[node_outputs[index]] = res[index]
        else:
            tf_tensor[node_outputs[0]] = res
    
    '''
        构建 Keras 模型:
    INFO: Keras 模型输入形状: (1, 256, 256, 4)
    INFO:keras_builder:Keras 模型输入形状: (1, 256, 256, 4)
    INFO: Keras 模型输出形状: (1, 512, 512, 3)
    INFO:keras_builder:Keras 模型输出形状: (1, 512, 512, 3)
    '''
    input_nodes = [tf_tensor[x.name] for x in model_graph.input] # 获取 ONNX 输入
    outputs_nodes = [tf_tensor[x.name] for x in model_graph.output] # 获取 ONNX 输出
    keras_model = keras.Model(inputs=input_nodes, outputs=outputs_nodes) # 构建 Keras 模型
    keras_model.trainable = False # 设定为不可训练
    # keras_model.summary() # 可选，打印模型结构
    # print(layout_dict)

####################################################################
    '''
        在返回模型之前，检查是否仍然包含动态输入/输出
    '''
    # 1. 获取 Keras 模型的输入输出形状
    input_shape = keras_model.input_shape
    output_shape = keras_model.output_shape

    # 2. 检查是否存在动态输入 (None 表示动态形状)
    if any(dim is None for dim in input_shape):
        LOG.warning(f"Keras 模型仍然包含动态输入: {input_shape}")

    # 3. 检查是否存在动态输出
    if any(dim is None for dim in output_shape):
        LOG.warning(f"Keras 模型仍然包含动态输出: {output_shape}")

    # 4. 记录信息
    LOG.info(f"Keras 模型输入形状: {input_shape}")
    LOG.info(f"Keras 模型输出形状: {output_shape}")

####################################################################
    # 记录 ONNX 的输入和输出布局
    input_layout, output_layout = {}, {}
    for inp in model_graph.input:
        input_layout[inp.name] = layout_dict[inp.name]
    for oup in model_graph.output:
        output_layout[oup.name] = layout_dict[oup.name]

    return keras_model, input_layout, output_layout # 返回 Keras 模型和布局信息


def tflite_builder(keras_model, weight_quant:bool=False, fp16_model=False, int8_model:bool=False, image_root:str=None,
                    int8_mean:list or float = [123.675, 116.28, 103.53], int8_std:list or float = [58.395, 57.12, 57.375]):
    
    """
    将 Keras 模型转换为 TFLite 模型，并支持不同的量化模式。

    参数:
    - keras_model (keras.Model): 需要转换的 Keras 模型。
    - weight_quant (bool, 可选): 是否进行权重量化。默认 False。
    - fp16_model (bool, 可选): 是否转换为 FP16 精度（适用于部分硬件优化）。默认 False。
    - int8_model (bool, 可选): 是否转换为 INT8 量化模型（适用于边缘设备）。默认 False。
    - image_root (str, 可选): 如果使用 INT8 量化，提供用于校准的图像数据目录。
    - int8_mean (list or float, 可选): INT8 量化校准的均值，默认 `[123.675, 116.28, 103.53]`。
    - int8_std (list or float, 可选): INT8 量化校准的标准差，默认 `[58.395, 57.12, 57.375]`。

    返回:
    - tflite_model (bytes): 转换后的 TFLite 模型。
    """
    # 1. 创建 TensorFlow Lite 转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # 2. 设定转换支持的运算                   # TensorFlow Lite 内置算子     # 允许使用部分 TensorFlow 原生算子
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    # 3. 启用量化选项
    if weight_quant or int8_model or fp16_model:
        converter.experimental_new_converter = True          # 使用新的 TFLite 转换器
        converter.optimizations = [tf.lite.Optimize.DEFAULT] # 启用优化

    # 4. 处理 FP16 量化（半精度浮点数）
    if fp16_model:
        converter.target_spec.supported_types = [tf.float16]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    # 5. 处理 INT8 量化
    elif int8_model:
        assert len(keras_model.inputs) == 1, f"help want, only support single input model."
        # 获取输入形状
        shape = list(keras_model.inputs[0].shape)
        # 选择数据集：使用 `image_root` 进行 INT8 量化校准
        dataset = RandomLoader(shape) if image_root is None else ImageLoader(image_root, shape, int8_mean, int8_std)
        # 设定代表性数据集（TFLite 量化需要一个校准数据集）
        converter.representative_dataset = lambda: dataset
        # 使用 INT8 计算
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.experimental_new_converter = True # 启用新的转换器
    # 6. 进行 TFLite 转换
    tflite_model = converter.convert()
    return tflite_model   # 返回 TFLite 模型