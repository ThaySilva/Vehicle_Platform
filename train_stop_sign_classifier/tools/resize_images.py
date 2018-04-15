import os
import sys
from PIL import Image

size = 250, 300
count = 596

for original_image in os.listdir("temp_negatives/"):
    image = Image.open("temp_negatives/"+original_image)
    image.thumbnail(size, Image.ANTIALIAS)
    image.save("negative_images/sample_negative_"+str(count)+".jpg","JPEG")
    count += 1
