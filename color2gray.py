import argparse
import matplotlib.pyplot as plt

from colorizers import *

from pathlib import Path
import os

from PIL import Image

# parser = argparse.ArgumentParser()
# parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
# parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
# parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
# opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()

imgdir = "Imagenet100/validation"
outputdir = f"Imagegray/validation"

c = 0

for folder in os.listdir(imgdir):
	c += 1
	folderpath = os.path.join(imgdir, folder)
	outputpath = os.path.join(outputdir, folder)
	x = 0
	if not os.path.exists(outputpath):
		os.makedirs(outputpath)
	for file in os.listdir(folderpath):
		x += 1
		print("processing", folder, x)
		filepath = os.path.join(folderpath, file)
		
		img = Image.open(filepath)
		bw = img.convert("L")
		f = file.replace(".JPEG", "")
		out = os.path.join(outputpath, f)
		bw.save("%s_gray.png"%out)
	print("Done class", c, folder)
