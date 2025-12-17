import onnx
from onnx_tf.backend import prepare
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

onnx_model_path = './LLE.onnx'
onnx_model = onnx.load(onnx_model_path)

onnx.checker.check_model(onnx_model)
print("ONNX to TensorFlow")

try:
    tf_rep = prepare(onnx_model)
    tf_model_path = 'lle_tf'
    tf_rep.export_graph(tf_model_path)
    print(f"Success, and save to {tf_model_path}")
except Exception as e:
    print(f"ERROR: {e}")

