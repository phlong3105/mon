from one.core import *
from one.constants import DATA_DIR

images_dir  = DATA_DIR / "inference" / "images"
outputs_dir = DATA_DIR / "inference" / "output"
images      = list(images_dir.rglob("*.jpg"))
outputs     = [str(i).replace("images", "output") for i in images]
outputs     = [str(o).replace(".jpg"  , ".txt")   for o in outputs]

for i, o in enumerate(outputs):
    image = cv2.imread(str(images[i]))
    
    with open(str(o), "r") as file:
        lines = file.read().splitlines()
        
        for l in lines:
            label, x1, y1, x2, y2, conf, _ = l.split(" ")
            x1   = float(x1)
            y1   = float(y1)
            x2   = float(x2)
            y2   = float(y2)
            conf = float(conf)
            if conf >= 0.9:
                start_point    = (int(x1), int(y1))
                end_point      = (int(x2), int(y2))
                color          = (0, 125, 255)
                line_thickness = 2
                image = cv2.rectangle(image, start_point, end_point, color, line_thickness)
        
        output_image = outputs_dir / images[i].name
        cv2.imwrite(str(output_image), image)
        cv2.imshow("Image", image)
        cv2.waitKey(100)
