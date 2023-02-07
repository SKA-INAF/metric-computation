# echo "trained on all, tested on all" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_all/all.json | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on all, tested on RGZ" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_all/rgz.json --subdataset RGZ | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on all, tested on ATCA" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_all/atca.json --subdataset ATCA | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on all, tested on ASKAP" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_all/askap.json --subdataset SCORPIO15 | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on RGZ, tested on all" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_rgz/all.json | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on RGZ, tested on RGZ" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_rgz/rgz.json --subdataset RGZ | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on RGZ, tested on ATCA" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_rgz/atca.json --subdataset ATCA | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on RGZ, tested on ASKAP" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_rgz/askap.json --subdataset SCORPIO15 | grep F1-Score | tail -n1 >> subsets.txt)

echo "trained on ATCA, tested on all" >> subsets.txt
$(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_atca/all.json | grep F1-Score | tail -n1 >> subsets.txt)

echo "trained on ATCA, tested on RGZ" >> subsets.txt
$(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_atca/rgz.json --subdataset RGZ | grep F1-Score | tail -n1 >> subsets.txt)

echo "trained on ATCA, tested on ATCA" >> subsets.txt
$(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_atca/atca.json --subdataset ATCA | grep F1-Score | tail -n1 >> subsets.txt)

echo "trained on ATCA, tested on ASKAP" >> subsets.txt
$(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_atca/askap.json --subdataset SCORPIO15 | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on ASKAP, tested on all" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_askap/all.json | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on ASKAP, tested on RGZ" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_askap/rgz.json --subdataset RGZ | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on ASKAP, tested on ATCA" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_askap/atca.json --subdataset ATCA | grep F1-Score | tail -n1 >> subsets.txt)

# echo "trained on ASKAP, tested on ASKAP" >> subsets.txt
# $(python main.py --data-dir yolov7@0.6 --gt-json gt_boxes.json --pred-json subsets/trained_on_askap/askap.json --subdataset SCORPIO15 | grep F1-Score | tail -n1 >> subsets.txt)
