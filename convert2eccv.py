import argparse
import matplotlib.pyplot as plt

from colorizers import *

from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(temperature=0.3).eval()

# imgpath = "BW_images/einstein.jpg"
# img = load_img(imgpath)
# (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
# img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
# out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
# out = "einstein"
# plt.imsave("%s_eccv_improved.png"%out, out_img_eccv16)

imgdir = "Imagenet100/validation"
outputdir = f"ECCV_modified/validation"

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
		
		img = load_img(filepath)
		(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
		f = file.replace(".JPEG", "")
		img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
		out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
		out = os.path.join(outputpath, f)
		plt.imsave("%s_eccv16.png"%out, out_img_eccv16)
	print("Done class", c, folder)
