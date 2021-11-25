'''
Converts COCO annotations in the format accepted by the notebook
'''

import json
import argparse

def main(args):
    with open(args.ann_file) as j:
        annotations = json.load(j)

    id_to_filename = {}
    for img in annotations['images']:
        id_to_filename[img['id']] = img['file_name'].split('.')[0]

    class_id_to_name = {}
    for cl in annotations['categories']:
        class_id_to_name[cl['id']] = cl['name']

    gt_boxes = {}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        img_name = id_to_filename[img_id]

        w, h = annotations['images'][img_id]['width'], annotations['images'][img_id]['height']

        if img_name not in gt_boxes:
            gt_boxes[img_name] = {'boxes': [], 'labels': []}
        bbox = ann['bbox'].copy()

        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        class_id = ann['category_id']
        class_name = class_id_to_name[class_id]
        gt_boxes[img_name]['boxes'].append(bbox)
        gt_boxes[img_name]['labels'].append(class_name)


    with open('gt_boxes.json', 'w') as out:
        json.dump(gt_boxes, out)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--ann_file", default="test.json", help="File with COCO annotations")

    # Optional argument flag which defaults to False
    parser.add_argument("--out", default="gt_boxes.json", help="Output JSON file")

    args = parser.parse_args()
    main(args)