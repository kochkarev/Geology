class_names = [
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

class_colors = [
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

lbls_to_colors = {name: class_colors[i] for i, name in enumerate(class_names)}
codes_to_colors = {i: class_colors[i] for i in range(len(class_names))}
