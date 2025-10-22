import os

import numpy as np
from PIL import Image
from tensorflow import lite


def load_model(model_path):
    interpreter = lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def print_model_structure(interpreter):
    print("Model Structure:")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nInput details:")
    for input_tensor in input_details:
        print(f"Name: {input_tensor['name']}, Shape: {input_tensor['shape']}, Type: {input_tensor['dtype']}")

    print("\nOutput details:")
    for output_tensor in output_details:
        print(f"Name: {output_tensor['name']}, Shape: {output_tensor['shape']}, Type: {output_tensor['dtype']}")


def preprocess_image(image_path):
    #target_size = (1024, 1024)
    img = Image.open(image_path).convert("RGB")  
    #img = img.resize(target_size, Image.BICUBIC)
    img_array = np.array(img).astype(np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    print(img_array.shape)
    return img_array


def inference_and_save_results(interpreter, input_image_folder, output_image_folder):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    for image_name in os.listdir(input_image_folder):
        image_path = os.path.join(input_image_folder, image_name)
        if not image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            continue  
        
        img_array = preprocess_image(image_path)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_img = interpreter.get_tensor(output_details[0]['index'])
        
        output_img = np.clip(output_img, 0., 1.)
        output_img = np.squeeze(output_img)
        output_img = (output_img * 255).astype(np.uint8)  

        pil_img = Image.fromarray(output_img)
        output_image_path = os.path.join(output_image_folder, image_name)
        pil_img.save(output_image_path)


def main():
    model_path = './LLE.tflite'  
    input_image_folder = './lowlight/LOLdataset/eval15/low'  
    output_image_folder = './experiments/results'  

    interpreter = load_model(model_path)
    print_model_structure(interpreter)
    inference_and_save_results(interpreter, input_image_folder, output_image_folder)


if __name__ == "__main__":
    main()
