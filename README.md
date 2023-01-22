# OCR Visualization Application


### Sample Results
<img src="./figures/text_detect_results.png">

*Hover over bounding boxes to display predicted text and confidence value*
<img src="./figures/text_predict_results.png">
<img src="./figures/filter_threshold_results.png">
<img src="./figures/inpainting_results.png">

## Getting Started
### Create Environment
```
conda create -n env python=3.10 -y
conda activate env
```
### Install dependencies
```
pip install -r requirements.txt
```
### Execute Application
```
python realtime_ocr_visualization.py
```