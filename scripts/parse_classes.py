def parse_classes():

	# import cv2 as cv
	import glob
	import json
	import os
	import numpy as np
	from PIL import Image

	proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	masks_path = os.path.join(proj_dir, "input", "UMNIK_2019", "BoxA_DS1", "masks_machine", "*.png")
	images = glob.glob(masks_path)

	result = dict()

	for image in images:
		# img = cv.imread(image)
		img = np.array(Image.open(image))
		result[image.split(os.sep)[-1]] = np.unique(img).tolist()

	with open(os.path.join(proj_dir, "input", "classes.json"), 'w') as fp:
		json.dump(result, fp, indent=4)

if __name__ == "__main__":
	parse_classes()