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


def bayer2rggb(img_bayer):
    h, w = img_bayer.shape
    img_bayer = img_bayer.reshape(h // 2, 2, w // 2, 2)
    img_bayer = img_bayer.transpose([1, 3, 0, 2]).reshape([4, h // 2, w // 2])  # [4, h//2, w//2]
    return img_bayer


def inference_and_save_results(interpreter, input_image_folder, output_image_folder):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    for image_name in os.listdir(input_image_folder):
        image_path = os.path.join(input_image_folder, image_name)
        if not image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            continue  
        
        img = Image.open(image_path)
        img_array = np.array(img)
        img_array = bayer2rggb(img_array)  # Convert Bayer pattern to RGGB
        img_array = img_array.astype(np.float32) / 4095.0  # Normalize to [0, 1]

        # Convert to shape [1, 128, 128, 4] for the model input
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = np.expand_dims(img_array, axis=0)  

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_img = interpreter.get_tensor(output_details[0]['index'])
        output_img = np.clip(output_img, 0., 1.)
        output_img = np.squeeze(output_img)
        print(output_img.shape)

        #output_img = output_img.transpose(1, 2, 0)
        output_img = (output_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(output_img)

        # Save the output image using PIL (Image.save)
        output_image_path = os.path.join(output_image_folder, image_name)
        pil_img.save(output_image_path)  # This automatically saves as RGB


def main():
    model_path = './ISP.tflite'  # TFLite model path
    input_image_folder = './ISP/Input'   # Input the Image folder path
    output_image_folder = './Output'  # Output the Image folder path

    interpreter = load_model(model_path)
    print_model_structure(interpreter)
    inference_and_save_results(interpreter, input_image_folder, output_image_folder)


if __name__ == "__main__":
    main()
