from ultralytics import YOLO

import mon

image_files = [
    "data/horses.jpg",
    "data/horses.jpg",
]
images  = [mon.read_image(image_file) for image_file in image_files]
kwargs  = {
    "conf": 0.10,
}
model   = YOLO(mon.ZOO_DIR / "vision/ultralytics/yolov8/yolov8n/coco/yolov8n_coco.pt")
results = model.predict(source=images, **kwargs)
for r in results:
    # print(r.orig_shape)
    # print(r)  # print detection bounding boxes
    print(r.boxes.data)
    # print(r.boxes.data)  # print detection bounding boxes
