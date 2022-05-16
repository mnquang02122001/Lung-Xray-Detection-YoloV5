# Smart diagnosis with Flask

## Introduction

With 1 lung X-ray image, we can detect 14 diseases of the lungs.

## Dataset

Data were obtained from VinBigData Chest X-ray Abnormalities Detection contest: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data

## Model

Yolov5 model (object detection model): https://github.com/ultralytics/yolov5

## Prerequisite

You must download required library: Flask, torch vision, numpy,...

## How to run this application?

- Clone this repository:

```bash
git clone https://github.com/mnquang02122001/Lung-Xray-Detection-YoloV5
```

- Start application:

```bash
cd yolov5
```

```bash
python svr_model.py
```

And you are ready to go!

By default, server runs on http://127.0.0.1:5000/

Enjoy!
