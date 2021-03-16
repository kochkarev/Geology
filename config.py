import os

classes_mask = {
    0 : "BG",
    1 : "Ccp",
    2 : "Gl",
    3 : "Mag",
    4 : "Brt",
    5 : "Po",
    6 : "PyMrc",
    7 : "Pn",
    8 : "Sph",
    9 : "Apy",
    10 : "Hem",
    11 : "TntTtr",
    12 : "Kvl"
}

classes_colors = {
    "BG" : "#000000",
    "Ccp" : "#ff0000",
    "Gl" : "#cbff00",
    "Mag" : "#00ff66",
    "Brt" : "#0065ff",
    "Po" : "#cc00ff",
    "PyMrc" : "#ff4c4c",
    "Pn" : "#dbff4c",
    "Sph" :	"#4cff93",
    "Apy" : "#4c93ff",
    "Hem" :	"#db4cff",
    "TntTtr" :	"#ff9999",
    "Kvl" :	"#eaff99"
}

classes_weights = {
    0: 0.7956840623086494,
    1: 0.768350642975015,
    2: 0.909053461587638,
    3: 2.9266693320289843
}

train_params = {
    "n_layers" : 4,
    "n_filters" : 16,
    "epochs" : 2,
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