from PIL import Image
import os

img_dir = 'spectrogram_images_'
folder = 500
width = 216
height = 216
i = 0
# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
for img in os.listdir(img_dir + str(folder) + "/"):
    pic = Image.open(img_dir + str(folder) + "/" + img)
# adjust width and height to your needs
    print(i+1, ": ", img)
    i = i+1
    im5 = pic.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    ext = ".png"
    im5.save(img_dir + str(width) + "/" + img[0:-4] + ext)

