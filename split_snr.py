from pathlib import Path
import argparse
import json

'''
Takes SNR split text files as a list of path of json files and returns the JSON annotation files
e.g. if the split folder has the following structure

- snr_split/test
| - less_5.txt
| - 5_to_20.txt
| - more_20.txt

the script will output two JSON files for each txt file, i.e. 

    pred_less_5.json
    gt_less_5.json

which will have the same structure as the one described in the notebook metric_computation.ipynb
'''

def parse_paths(txt_path):
    with open(txt_path) as txt:
        path_list = list(txt)

    parsed_list = []
    for filename in path_list:
        file_path = Path(filename)
        img_name = file_path.stem.replace('_mask', '')
        parsed_list.append(img_name)
    
    return parsed_list

def main(args):
    with open(args.data_dir / args.gt_json) as infile:
        gt_boxes = json.load(infile)

    with open(args.data_dir / args.pred_json) as infile:
        pred_boxes = json.load(infile)

    split_gt_boxes = {}
    split_pred_boxes = {}

    snr_split_path = Path(args.split_folder)
    for split in snr_split_path.glob('*'):

        split_gt_boxes = {}
        split_pred_boxes = {}

        img_names = parse_paths(split)
        for img_name in img_names:

            if img_name in gt_boxes:
                split_gt_boxes[img_name] = gt_boxes[img_name]
            
            if img_name in pred_boxes:
                split_pred_boxes[img_name] = pred_boxes[img_name]

        with open(args.data_dir / f'gt_{split.stem}.json', 'w') as out:
            json.dump(split_gt_boxes, out)
        with open(args.data_dir / f'pred_{split.stem}.json', 'w') as out:
            json.dump(split_pred_boxes, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("--data_dir", default="sample_jsons", help="Directory of JSON files")
    parser.add_argument("--gt_json", default="gt_boxes.json", help="JSON file with ground truth annotations")
    parser.add_argument("--pred_json", default="pred_boxes.json", help="JSON file with prediction annotations")
    parser.add_argument("--split_folder", default="snr_splits", help="Folder with SNR based split")

    # Optional argument flag which defaults to False

    args = parser.parse_args()

    args.data_dir = Path(args.data_dir)

    main(args)