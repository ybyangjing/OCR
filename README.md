# MSCL-OCR: Multiscale Spatial-Aware Collaborative Learning Framework for Optical Character Recognition in Instruments and Gauges



## ⚙️ Requirements

Install all required dependencies into a new virtual environment via conda.

- Python >= 3.8

- PyTorch >= 1.8

- torchvision

- numpy

- opencv-python

  Install dependencies using:
  ```bash
  pip install -r requirements.txt

## 📊 Datasets

**Bublic_datasets:**

- ADI: https://aistudio.baidu.com/datasetdetail/215211
- YUVA EB：https://data.mendeley.com/datasets/fnn44p4mj8/1?utm_source=chatgpt.com

## 🚀 Training

Please configure the paths for the training/test/validation datasets in the `configs/data.yaml` file before running the `scripts/train.py` file to begin training.

```python
python3 scripts/train.py
```

## 📈 Evaluation

After training, evaluate the model using:	

```python
python scripts/evaluate.py --config ./configs --checkpoint path/to/checkpoint.pth --ann_path path/to/annotations.json --img_dir path/to/images --output_dir path/to/output
```

Evaluation metrics include:

- Recognition Accuracy
- Character Error Rate (CER)



If you have any questions you can contact us: jyang23@gzu.edu.cn 
