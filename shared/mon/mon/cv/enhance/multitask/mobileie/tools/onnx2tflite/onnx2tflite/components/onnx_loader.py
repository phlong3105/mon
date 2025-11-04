import os
import onnx
import logging
from onnxsim import simplify # ONNX 模型简化工具

LOG = logging.getLogger("onnx_loader running:")
LOG.setLevel(logging.INFO)

def clean_model_input(model_proto):
    """
    清理 ONNX 模型的输入，删除 ONNX 计算图中冗余的输入节点。

    逻辑：
    - 遍历 ONNX 计算图中的 `graph.input`
    - 如果某个 `input` 也出现在 `initializer` 中，则说明它是一个冗余输入（即它的值已经在 `initializer` 中存储）
    - 从 `graph.input` 中移除这些冗余输入

    参数：
    - model_proto (onnx.ModelProto): 需要清理的 ONNX 模型
    """
    inputs = model_proto.graph.input # 获取 ONNX 计算图中的输入
    name_to_input = {}   # 创建输入名称到输入对象的映射
    for input in inputs:
        name_to_input[input.name] = input

    names = []
    for initializer in model_proto.graph.initializer: # 遍历所有初始化参数
        if initializer.name in name_to_input:         # 如果初始化参数的名字在输入列表中
            inputs.remove(name_to_input[initializer.name]) # 删除该输入
            names.append(initializer.name)
    
    if len(names) > 0:
        LOG.warning(f"[{len(names)}] redundant input nodes are removed.\n \
            nodes name : {','.join(names)}")

def get_onnx_submodel(onnx_model_path:str, input_node_names:list=None, output_node_names:list=None):
    """
    截取 ONNX 子模型，即从原始 ONNX 模型中提取以 `input_node_names` 为输入、
    `output_node_names` 为输出的子图。

    逻辑：
    - 载入 ONNX 模型
    - 确定输入节点和输出节点（如果未指定，则默认使用整个模型的输入/输出）
    - 使用 `onnx.utils.extract_model` 提取子模型并保存
    - 载入提取后的子模型并返回

    参数：
    - onnx_model_path (str): ONNX 模型文件路径
    - input_node_names (list, optional): 指定子模型的输入节点名称
    - output_node_names (list, optional): 指定子模型的输出节点名称

    返回：
    - model_proto (onnx.ModelProto): 提取的子模型
    """
    model_proto = onnx.load(onnx_model_path) # 载入 ONNX 模型
    # 如果未指定输入节点，则默认使用 ONNX 模型的全部输入
    if input_node_names is None:
        input_node_names = []
        for inp in model_proto.graph.input:
            input_node_names.append(inp.name)
    
    # 如果未指定输出节点，则默认使用 ONNX 模型的全部输出
    if output_node_names is None:
        output_node_names = []
        for oup in model_proto.graph.output:
            output_node_names.append(oup.name)
    del model_proto # 释放原始模型的内存
    
    # 生成新模型的文件路径
    new_model_path = os.path.splitext(onnx_model_path)[0] + "_sub.onnx"
    # 提取子模型并保存
    onnx.utils.extract_model(onnx_model_path, new_model_path, input_node_names, output_node_names)
    # 载入提取后的子模型
    model_proto = onnx.load(new_model_path)
    return model_proto

def get_proto(onnx_model_path:str, input_node_names:list=None, output_node_names:list=None):
    if input_node_names is None and output_node_names is None:
        return onnx.load(onnx_model_path)
    else:
        return get_onnx_submodel(onnx_model_path, input_node_names, output_node_names)
    
def load_onnx_modelproto(onnx_model_path:str, input_node_names:list=None, output_node_names:list=None, need_simplify:bool=True):
    """
    载入 ONNX 模型，并根据需要进行简化和清理。

    逻辑：
    - 检查 ONNX 模型文件是否存在
    - 载入完整的 ONNX 模型或子模型
    - 检测是否存在动态输入
    - 如果 `need_simplify=True`，则尝试使用 `onnx-simplifier` 进行模型优化
    - 移除 ONNX 计算图中冗余的输入

    参数：
    - onnx_model_path (str): ONNX 模型文件路径
    - input_node_names (list, optional): 需要提取的输入节点
    - output_node_names (list, optional): 需要提取的输出节点
    - need_simplify (bool, optional): 是否对 ONNX 进行简化，默认启用

    返回：
    - model_proto (onnx.ModelProto): 处理后的 ONNX 模型
    """
    # 1. 检查 ONNX 文件是否存在
    if not os.path.exists(onnx_model_path):
        LOG.error(f"{onnx_model_path} is not exists.")
        raise FileExistsError(f"{onnx_model_path} is not exists.")
    # 2. 载入 ONNX 模型或子模型
    model_proto = get_proto(onnx_model_path, input_node_names, output_node_names)
    # 3. 检查是否存在动态输入（即输入形状中有未指定的维度）
    dynamic_input = False
    for inp in model_proto.graph.input:
        for x in inp.type.tensor_type.shape.dim:
            if x.dim_value <= 0:  # 发现动态输入
                dynamic_input = True
                break
    # 4. 进行 ONNX 模型简化（如果启用）        
    if need_simplify:
        success = False
        try:
            # 使用 `onnxsim` 进行模型优化，允许动态输入
            model_proto, success = simplify(model_proto, check_n=1, dynamic_input_shape=dynamic_input)
        except:
            success = False
        # 如果简化失败，记录警告信息
        if not success:
            LOG.warning(f"onnxsim is failed, maybe make convert fails.")

            model_proto = onnx.load(onnx_model_path)
            
        # 5. 清理 ONNX 模型的冗余输入
        clean_model_input(model_proto)
        # 在返回 ONNX 之前，检查是否仍然存在动态输入

##################################################################################
    for inp in model_proto.graph.input:
        for x in inp.type.tensor_type.shape.dim:
            if x.dim_value <= 0:  # 仍然是动态输入
                LOG.warning(f"ONNX 仍然包含动态输入: {inp.name}，维度: {[dim.dim_value for dim in inp.type.tensor_type.shape.dim]}")
                break  # 只打印一次警告即可
##################################################################################

    return model_proto