import os

classes_mask_all = [
    "BG",
    "Ccp",
    "Gl",
    "Mag",
    "Brt",
    "Po",
    "Py/Mrc",
    "Pn",
    "Sph",
    "Apy",
    "Hem",
    "Tnt/Ttr",
    "Kvl",
]

classes_mask = [
    'BG',
    'Ccp',
    'Gl',
    'Brt',
    'Py/Mrc',
    'Sph',
    'Tnt/Ttr',
]

# classes_mask = [
#     'BG',
#     'other',
#     'Gl',
#     'other',
#     'Py/Mrc',
#     'Sph',
#     'other',
# ]

classes_colors_all = [
    '#000000',
    '#ff0000',
    '#cbff00',
    '#00ff66',
    '#0065ff',
    '#cc00ff',
    '#ff4c4c',
    '#dbff4c',
    '#4cff93',
    '#4c93ff',
    '#db4cff',
    '#ff9999',
    '#eaff99',
]

classes_colors = [
    '#000000',
    '#ff0000',
    '#cbff00',
    '#0065ff',
    '#ff4c4c',
    '#4cff93',
    '#ff9999'
]

# classes_weights = {
#     0: 0.7956840623086494,
#     1: 0.768350642975015,
#     2: 0.909053461587638,
#     3: 2.9266693320289843
# }

train_params = {
    "n_layers" : 4,
    "n_filters" : 16,
    "epochs" : 100,
    "aug_factor" : 5,
    "dataset_path" : os.path.join('input', 'dataset'),
    "batch_size" : 32,
    "patch_size" : 256,
    "overlay" : 0.25,
    "output_path" : "output",
    "heatmaps_input" : os.path.join('input', 'heatmaps'),
    "heatmaps_output" : os.path.join('test_output', "heatmaps"),
    "ores_json" : os.path.join('input', 'ores.json'),
    "full_augment" : False,
    "model_path" : os.path.join("models")
}