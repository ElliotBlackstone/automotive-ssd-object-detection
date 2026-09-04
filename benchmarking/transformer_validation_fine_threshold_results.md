# Transformer validation fine-threshold sweep

- Training directory: `C:\Udacity_car_data\data\train`
- Validation split: 25%, seed 724 (4967 images evaluated)
- Matching: class-aware, one-to-one, IoU >= 0.50

| Confidence threshold | Precision | Recall | F1 | False positives / image |
|---:|---:|---:|---:|---:|
| 0.95 | 0.8189 | 0.7625 | 0.7897 | 1.0964 |
| 0.96 | 0.8367 | 0.7393 | 0.7850 | 0.9386 |
| 0.97 | 0.8601 | 0.7055 | 0.7752 | 0.7461 |
| 0.98 | 0.8885 | 0.6435 | 0.7464 | 0.5249 |
| 0.985 | 0.9072 | 0.5962 | 0.7196 | 0.3966 |
| 0.99 | 0.9282 | 0.5312 | 0.6757 | 0.2674 |
| 0.9925 | 0.9437 | 0.4865 | 0.6420 | 0.1888 |
| 0.995 | 0.9583 | 0.4253 | 0.5891 | 0.1202 |
| 0.9975 | 0.9784 | 0.3289 | 0.4922 | 0.0473 |
| 0.999 | 0.9912 | 0.2278 | 0.3705 | 0.0131 |
