from ultralytics import YOLO


def train():
    model = YOLO('yolo11n.pt')

    results = model.train(data='datasets/dataset.yaml', epochs=2)
    print(results)

    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # Perform object detection on an image using the model
    results = model("short_close_range.mp4")
    print(results)
    # Export the model to ONNX format
    success = model.export(format="onnx")

    print(success)

if __name__ == '__main__':
    train()