from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generate_binary_structure, label
from skimage.measure import regionprops
from tensorflow.keras.utils import to_categorical
import json
import os
from config import classes_mask

class SegmentationStat:
    
    @staticmethod
    def calculate_stat(image: np.ndarray, save_path: str = None):
        stat = {}

        stat['ores_distr'] = SegmentationStat.__ores_distr(image)
        # print(stat['ores_distr'])

        stat['ores_area'] = SegmentationStat.__ores_area(image)
        # print(stat['ores_area'])

        # stat['ores_instance_area_distr'] = SegmentationStat.__ores_instance_area_distr(image)
        # print(stat['ores_instance_area_distr'])

        stat['ores_instance_distr'] = SegmentationStat.__calc_props(image)

        if save_path:
            with open(os.path.join(save_path, "segm_stat.json"), 'w') as fp:
                json.dump(stat, fp, indent=4)

    @staticmethod
    def __ores_distr(image: np.ndarray):
        classes = np.unique(image)
        result = {}
        for class_id in classes:
            result[classes_mask[class_id]] = np.count_nonzero(image == class_id) / (image.shape[0] * image.shape[1]) * 100
        return result

    @staticmethod
    def __ores_area(image: np.ndarray, pixel_lenght_mm:int = 0.0004):
        k = pixel_lenght_mm ** 2
        classes = np.unique(image)
        result = {}
        for class_id in classes:
            result[classes_mask[class_id]] = np.count_nonzero(image == class_id) * k
        return result

    @staticmethod
    def __ores_instance_distr(image: np.ndarray):
        n_classes = len(classes_mask)
        image_classes = to_categorical(image, num_classes=n_classes)
        result = {}

        for i in range(n_classes):
            mask = image_classes[...,i]
            areas = []
            labeled, num_features = label(input=mask, structure=generate_binary_structure(2,2))

            for j in range(1, num_features + 1):
                areas.append(np.count_nonzero(labeled == j))

            result[classes_mask[i]] = {"instances_num" : str(len(areas)), "average_area" : str(np.mean(areas))}    
            # result[classes_mask[i]] = np.histogram(areas, bins='auto')
            # plt.hist(areas, bins='auto')
            # plt.savefig(classes_mask[i] + ".png")

        return result

    @staticmethod
    def __calc_props(image: np.ndarray):
        n_classes = len(classes_mask)
        image_classes = to_categorical(image, num_classes=n_classes)
        result = {}

        for class_id in range(n_classes):
            stat = []
            mask = image_classes[...,class_id]
            labeled, num_features = label(input=mask, structure=generate_binary_structure(2,2))
            props = regionprops(labeled)

            for prop in props:
                stat.append({"area" : str(prop.area), "perimeter" : str(prop.perimeter), "ratio" : str(prop.perimeter / prop.area)})
            
            result[classes_mask[class_id]] = stat

        return result


if __name__ == '__main__':

    image = np.array(Image.open('input/dataset/Py-Cpy-Sh-BR-GL31_NEW.png'))[...,0]

    SegmentationStat.calculate_stat(image, '/home/akochkarev/geology')