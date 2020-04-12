import os
from utils import create_heatmaps
from data_utils import generate_lists_mineral

if __name__ == "__main__":

    # Generating heatmaps
    output_path=os.path.join('input', 'heatmaps')
    input_path=os.path.join('input', 'dataset')
    vis_path = os.path.join('test_output', 'heatmaps')

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.makedirs(vis_path, exist_ok=True)
        create_heatmaps(
            num_classes=4,
            patch_size=512,
            input_path=input_path,
            output_path=output_path,
            vis_path=vis_path,
            visualize=True
        )
    else:
        print(f'Skipping heatmaps generation. To regenerate delete directory {output_path}')

    # Generating weighted list of images
    output_path = os.path.join('input', 'ores.json')
    if not os.path.exists(output_path):
        print('Generating weighted list for all ores')
        generate_lists_mineral(input_path=input_path, output_path=output_path)
        print(f'Weighted list saved in {os.path.join(input_path, "ores.json")}')
    else:
        print(f'Skipping generation of weighted list for ores. To regenerate delete file {output_path}')