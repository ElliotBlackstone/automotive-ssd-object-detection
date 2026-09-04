# Test-set detector comparison

- Test set: `C:\Udacity_car_data\data\test` (9937 images)
- SSD threshold: 0.300
- Transformer threshold: 0.950
- Threshold matching: class-aware, one-to-one, IoU >= 0.50
- SSD AP score floor: 0.050
- AP protocol: COCO interpolation at IoU 0.50:0.05:0.95, maxDets=100; size categories use native-image object areas

| Metric | SSD v2 | Transformer |
|---|---:|---:|
| mAP@0.50 | 0.5697 | 0.7310 |
| mAP@0.75 | 0.2778 | 0.3789 |
| mAP@[0.50:0.95] | 0.3042 | 0.3934 |
| AP small | 0.1848 | 0.3087 |
| AP medium | 0.4109 | 0.5084 |
| AP large | 0.6847 | 0.6332 |
| AP biker | 0.2008 | 0.3088 |
| AP car | 0.4455 | 0.5155 |
| AP pedestrian | 0.1266 | 0.2100 |
| AP traffic light | 0.2837 | 0.4043 |
| AP truck | 0.4647 | 0.5285 |
| Precision @ chosen threshold | 0.8341 | 0.8176 |
| Recall @ chosen threshold | 0.6710 | 0.7634 |
| F1 @ chosen threshold | 0.7438 | 0.7896 |
| False positives / image | 0.8752 | 1.1168 |
| Mean IoU of true positives | 0.7884 | 0.7953 |
