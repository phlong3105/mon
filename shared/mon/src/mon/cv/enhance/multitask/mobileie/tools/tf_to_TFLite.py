import os

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

saved_model_dir = "./lle_tf" 
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

tflite_model_path = "LLE.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite save to {tflite_model_path}")
