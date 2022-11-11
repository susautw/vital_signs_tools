import json
import shutil
from pathlib import Path

import numpy as np

root = Path("ds3_yolo_2_noise_removed")

choices = ["train", "val", "test"]
p = [0.6, 0.2, 0.2]

class_name_id_map = {
    "metronome": 0
}


def main():
    # read all image path
    for image_path in sorted((root / "images").glob("*.png")):
        # find corresponding label of the image
        label_path = root / "labels" / f'{image_path.stem}.json'
        # choose where to store
        dest_root = root / np.random.choice(choices, p=p)
        dest_images = dest_root / "images"
        dest_labels = dest_root / "labels"

        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        # covert labelme to yolo format and store it
        with label_path.open() as fp:
            label_context = json.load(fp)
        factor = [label_context["imageWidth"], label_context["imageHeight"]] * 2
        rects = []
        for shape in label_context["shapes"]:
            if shape["shape_type"] != "rectangle":
                continue
            (x1, y1), (x2, y2) = shape["points"]
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            rects.append(
                [class_name_id_map[shape['label']]] + list(np.array([xc, yc, w, h]) / factor)
            )
        shutil.copy2(image_path, dest_images)
        with (dest_labels / f'{image_path.stem}.txt').open("w") as fp:
            fp.writelines([rect_to_str(rect) for rect in rects])


def rect_to_str(rect):
    return ' '.join(map(str, rect))


if __name__ == '__main__':
    main()
