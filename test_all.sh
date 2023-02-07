#!/bin/bash
subdataset=$1

echo "detr" >> $subdataset.txt
$(python main.py --data-dir detr --gt-json gt_boxes.json --pred-json rg-boxes.json --subdataset $subdataset | grep F1-Score | tail -n1 >> $subdataset.txt)

echo "detectron2" >> $subdataset.txt
$(python main.py --data-dir detectron2 --gt-json gt_boxes.json --pred-json pred_boxes.json --subdataset $subdataset | grep F1-Score | tail -n1 >> $subdataset.txt)

echo "effdet-d1" >> $subdataset.txt
$(python main.py --data-dir effdet/d1 --gt-json gt_boxes.json --pred-json pred_boxes@0.9_formatted.json --subdataset $subdataset | grep F1-Score | tail -n1 >> $subdataset.txt)

echo "effdet-d2" >> $subdataset.txt
$(python main.py --data-dir effdet/d2 --gt-json gt_boxes.json --pred-json pred_boxes@0.9_formatted.json --subdataset $subdataset | grep F1-Score | tail -n1 >> $subdataset.txt)

echo "yolov4" >> $subdataset.txt
$(python main.py --data-dir yolo@0.75 --gt-json gt_boxes.json --pred-json pred_boxes.json --subdataset $subdataset | grep F1-Score | tail -n1 >> $subdataset.txt)

echo "yolov7" >> $subdataset.txt
$(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json pred_boxes.json --subdataset $subdataset | grep F1-Score | tail -n1 >> $subdataset.txt)