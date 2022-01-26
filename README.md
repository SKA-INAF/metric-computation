## Predictions Format

The model accepts predictions as JSON files in the following format
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

## Ground Truth Format

Same as before, without the scores
```json
    {
        "$IMG_NAME": {
            "labels": [$CLASS_NAME: string, ..., $CLASS_NAME: string],
            "boxes": [[$X1, $X2, $Y1, $Y2], ..., [$X1, $X2, $Y1, $Y2]],
        },

        ...
        
        "$IMG_NAME": {
            "labels": [$CLASS_NAME: string, ..., $CLASS_NAME: string],
            "boxes": [[$X1, $X2, $Y1, $Y2], ..., [$X1, $X2, $Y1, $Y2]],
        }
    }
```

### `from_coco`
Used to convert annotations in COCO format to the format accepted by the notebook