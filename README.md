# Metric Computation
This repo contains several scripts to compute object detection metrics using two different type of scripts: one uses a custom implementation from Mask R-CNN analysis script, the other one uses torchmetrics functions.



## Data format

The model accepts predictions and ground truth data as JSON files.
For **predictions**, data must follow the following format:
(The notation in caps with a dollar sign (`$IMG_NAME`) means that such key will be replaced by the actual value of the property)
```json
    {
        "$IMG_NAME": {
            "labels": [$CLASS_NAME: string, ..., $CLASS_NAME: string],
            "boxes": [[$X1, $X2, $Y1, $Y2], ..., [$X1, $X2, $Y1, $Y2]],
            "scores": [$CONFIDENCE_SCORE: float, ..., $CONFIDENCE_SCORE: float]
        },

        ...
        
        "$IMG_NAME": {
            "labels": [$CLASS_NAME: string, ..., $CLASS_NAME: string],
            "boxes": [[$X1, $X2, $Y1, $Y2], ..., [$X1, $X2, $Y1, $Y2]],
            "scores": [$CONFIDENCE_SCORE: float, ..., $CONFIDENCE_SCORE: float]
        }
    }
```
Bounding box coordinates are to be intended in the following way:
    `X1, Y1`: coordinates of the upper left point of the box;
    `X2, Y2`: coordinates of the bottom right point of the box;

For **ground truth**, the format is the same but the *scores* key under each image is missing as it's not used.

## Data preparation
In order to run the script, both JSON files has to be provided to the `main.py` script and put into what will be passed as argument as `--data_dir`. Output will be stored in the same folder.

The script `from_coco.py` allows for converting data in the required format starting from the common COCO annotation format, which is also in JSON.

## Run the script
    ``` bash
    python main.py --data_dir {path/to/folder_containing_jsons} --gt_json {gt_filename} --pred_json {pred_filename}
    ```

The script `split_snr.py` is used to generate predictions and ground truth files according to the SNR value splits, which are located in the folder `snr_splits`. The arguments to pass are the same, except for `--split_folder` which indicates the snr splits txt files on which the subdivision is based.
    ``` bash
    python split_snr.py --data_dir {path/to/folder_containing_jsons} --gt_json {gt_filename} --pred_json {pred_filename} --split_folder snr_splits
    ```