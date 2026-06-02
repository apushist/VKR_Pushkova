
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolo26n.pt") 

    results = model.train(
        data="data.yaml",
        imgsz = 640,
        batch = 16,
        device = 0,
        exist_ok=True,
        name="final",
        epochs = 80,
        patience=15,
        warmup_epochs = 4,
        seed = 42,
        verbose = False,
        scale = 0.1,
        box = 12.0,
        cos_lr = True,
        hsv_v = 0.5,
        weight_decay = 3e-4,
        fliplr=0.0,     
        cls=1.0, 
        )

