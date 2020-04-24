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

train_params = {
    "n_layers" : 3,
    "n_filters" : 16,
    "epochs" : 100,
    "dataset_path" : os.path.join('input', 'dataset'),
    "batch_size" : 32,
    "patch_size" : 256,
    "overlay" : 0.25,
    "output_path" : "output",
    "heatmaps_input" : os.path.join('input', 'heatmaps'),
    "heatmaps_output" : os.path.join('test_output', "heatmaps"),
    "ores_json" : os.path.join('input', 'ores.json')
}