import os
import shutil
import fnmatch
import xlrd
import cv2 as cv
import json

def parse_dataset(path):
    
    result = {}

    wb = xlrd.open_workbook(path)
    sheet_num = len(wb.sheet_names())

    for sheet_idx in range(1, sheet_num):

        sheet = wb.sheet_by_index(sheet_idx)
        sheet.cell_value(0, 0)
        dataset_name = (sheet.cell_value(0, 0)).split()[0]

        marked = []
        train = []
        test = []

        for i in range(sheet.nrows):
            if type(sheet.cell_value(i, 0)) == type("aaa") and (sheet.cell_value(i, 0)).endswith(".jpg"):
                marked.append(sheet.cell_value(i, 0))
            if type(sheet.cell_value(i, 1)) == type("aaa") and (sheet.cell_value(i, 1)).endswith(".jpg"):
                train.append(sheet.cell_value(i, 1))
            if type(sheet.cell_value(i, 2)) == type("aaa") and (sheet.cell_value(i, 2)).endswith(".jpg"):
                test.append(sheet.cell_value(i, 2))

        result[dataset_name] = {"marked" : marked, "train" : train, "test" : test}

    with open(os.path.join("input", "dataset.json"), 'w') as fp:
        json.dump(result, fp, indent=4)

def make_dataset():

    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    classes_path = os.path.join(proj_dir, 'input', 'UMNIK_2019', 'obj_class_to_machine_color.json')

    Background_val = 0
    Sh_new_val = 1
    PyMrc_new_val = 2
    Gl_new_val = 3

    with open(classes_path) as classes_json:
        data1 = json.load(classes_json)
    data = {"Sh" : [[], Sh_new_val], "PyMrc" : [[], PyMrc_new_val], "Gl" : [[], Gl_new_val], "Other" : [[0], Background_val]}
    for elem in data1.keys():
        if elem.startswith('S'):
            data["Sh"][0].append(data1[elem][0])
        elif elem.startswith('P'):
            data["PyMrc"][0].append(data1[elem][0])
        elif elem.startswith('Gl'):
            data["Gl"][0].append(data1[elem][0])
        else:
            data["Other"][0].append(data1[elem][0])

    data_values = dict()
    for elem in data.keys():
        for val in data[elem][0]:
            data_values[val] = elem

    def process_mask(mask):
        print('Processing: ' + mask)
        img = cv.imread(mask)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_val = data[data_values[img[i, j, 0]]][1]
                img[i, j, :] = (new_val, new_val, new_val)
        cv.imwrite(mask.replace(".png","_NEW.png"), img)

    dataset_dir = os.path.join(proj_dir, 'input', 'dataset')
    path = os.path.join(proj_dir, 'input', 'UMNIK_2019', 'BoxA_DS1')

    DIR = os.path.join(path, 'img')
    dataset_size = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    try:
        os.mkdir(dataset_dir)
    except FileExistsError:
        shutil.rmtree(dataset_dir)
        os.mkdir(dataset_dir)

    names = []

    inp = os.path.join(proj_dir, 'input', 'OreGeology-Dateset.xlsx')
    wb = xlrd.open_workbook(inp) 
    sheet = wb.sheet_by_index(1)
    sheet.cell_value(0, 0) 
    for i in range(sheet.nrows):
        if type(sheet.cell_value(i, 0)) == type("aaa"):
            names.append(sheet.cell_value(i, 0))

    names = list(dict.fromkeys(names))

    marked_images = 0
    for fname in names:
        if fname.endswith('.jpg'):
            for file_name in os.listdir(os.path.join(path, 'img')):
                if file_name.startswith(fname[:fname.find('.')]):    
                    shutil.copy(os.path.join(path, 'img', file_name), dataset_dir)
            for file_name in os.listdir(os.path.join(path, 'masks_machine')):
                if file_name.startswith(fname[:fname.find('.')]):    
                    shutil.copy(os.path.join(path, 'masks_machine', file_name), dataset_dir)
                    process_mask(os.path.join(dataset_dir, file_name))
            marked_images += 1

    print("{} marked images in dataset of {} images".format(marked_images, dataset_size))

if __name__ == "__main__":
    make_dataset()