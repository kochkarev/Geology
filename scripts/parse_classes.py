def parse_classes():

	import cv2 as cv
	import glob
	import json
	import os

	proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	masks_path = os.path.join(proj_dir, "input", "UMNIK_2019", "BoxA_DS1", "masks_machine", "*.png")
	images = glob.glob(masks_path)

	result = dict()

	for image in images:
		img = cv.imread(image)
		d = dict()
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				d[str(img[i, j, 0])] = 5
		result[image.split(os.sep)[-1]] = [key for key in d.keys()]

	with open(os.path.join(proj_dir, "input", "classes.json"), 'w') as fp:
		json.dump(result, fp)

if __name__ == "__main__":
	parse_classes()