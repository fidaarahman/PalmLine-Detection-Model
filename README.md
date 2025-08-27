### Palm Line Detection Model

This repository contains a YOLOv8-based model for detecting palm lines from hand images.

The project is trained using a custom dataset and can be used for palm line recognition and analysis.

### 🚀 Features

- **Palm line detection using YOLOv8**
- **Custom dataset prepared and trained with Roboflow**
- **Pre-trained weights available for direct use**
- **Easy to train, test, and evaluate**

### 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/fidaarahman/PalmLine-Detection-Model.git
cd PalmLine-Detection-Model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 📊 Training

To train the model, run:

```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640
```

### 🔍 Inference

To detect palm lines on a custom image:

```bash
yolo task=detect mode=predict model=best.pt source="path/to/your/image.jpg"
```

### 🖼️ Results

<img width="322" height="403" alt="image" src="https://github.com/user-attachments/assets/fef847b1-3e63-47a5-8665-72fe5d29ad5a" />


### 📥 Model Weights

You can download the trained model weights from Google Drive:

[Download Model](https://drive.google.com/your-model-link-here)


### 🙌 Acknowledgements

- **Ultralytics YOLOv8**
- **Roboflow for dataset preparation**

### 👤 Author

- **Name**: Fida ur Rahman
- **Email**: fidaurrahman700@gmail.com
- **LinkedIn**: www.linkedin.com/in/fidarh24

