# fast-sam-nvidia-jeston

Step 1: setup jetson

pip install -r requirements.txt

pip install git+https://github.com/openai/CLIP.git

Download model weights in PyTorch format

convert to TensorRT
Create a new Python script and enter the following code. Save and execute the file.

from ultralytics import YOLO


model = YOLO('FastSAM-s.pt')  # load a custom trained
# TensorRT FP32 export
# model.export(format='engine', device='0', imgsz=640)
# TensorRT FP16 export
model.export(format='engine', device='0', imgsz=640, half=True)


python3 Inference_video.py --model_path <path to model> --input_path <id of camera> --imgsz 640
