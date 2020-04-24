import os
from utils import create_heatmaps
from data_utils import generate_lists_mineral
from scripts.make_dataset import make_dataset
from config import train_params, classes_mask

if __name__ == "__main__":

    # Creating dataset
    input_path=train_params["dataset_path"]
    if not os.path.exists(input_path):
        make_dataset()
    else:
        print(f'Skipping dataset generation. To regenerate delete directory {input_path}')

    # Generating heatmaps
    output_path = train_params["heatmaps_input"]
    vis_path = train_params["heatmaps_output"]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.makedirs(vis_path, exist_ok=True)
        create_heatmaps(
            num_classes=len(classes_mask.keys()),
            patch_size=train_params["patch_size"],
            input_path=input_path,
            output_path=output_path,
            vis_path=vis_path,
            visualize=True
        )
    else:
        print(f'Skipping heatmaps generation. To regenerate delete directory {output_path}')

    # Generating weighted list of images
    output_json = train_params["ores_json"]
    if not os.path.exists(output_json):
        print('Generating weighted list for all ores')
        generate_lists_mineral(input_path=input_path, output_path=output_json)
        print(f'Weighted list saved in {output_json}')
    else:
        print(f'Skipping generation of weighted list for ores. To regenerate delete file {output_json}')