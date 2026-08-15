from PIL import Image
import os

images_path = "data/raw/RIVA/riva_1.0/images"

files = os.listdir(images_path)

print("Number of images:", len(files))

for filename in files[:10]:
    path = os.path.join(images_path, filename)

    try:
        with Image.open(path) as img:
            print(
                filename,
                "->",
                "width:", img.width,
                "height:", img.height,
                "mode:", img.mode
            )
    except Exception as e:
        print(filename, "ERROR:", e)