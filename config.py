import os

classes_mask = {
    0 : "Other",
    1 : "Sh",
    2 : "PyMrc",
    3 : "Gl"
}

classes_colors = {
    "Other" : "#000000",
    "Sh" : "#FFA500",
    "PyMrc" : "#F08080",
    "Gl" : "#20B2AA",
    "Tnt" : "#FF69B4",
    "Cch" : "#ADFF2F",
    "Br" : "#4B0082"
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
    "epochs" : 100,
    "aug_factor" : 7,
    "dataset_path" : os.path.join('input', 'dataset'),
    "batch_size" : 32,
    "patch_size" : 256,
    "overlay" : 0.25,
    "output_path" : "output",
    "heatmaps_input" : os.path.join('input', 'heatmaps'),
    "heatmaps_output" : os.path.join('test_output', "heatmaps"),
    "ores_json" : os.path.join('input', 'ores.json'),
    "full_augment" : True,
    "model_path" : os.path.join("models")
}