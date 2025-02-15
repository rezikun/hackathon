from ultralytics import YOLO


def train():
    model = YOLO('yolo11s.pt')

    results = model.train(
      data='datasets/dataset.yaml', 
      epochs=200, device=0, 
      save_period=5, 
      project="/content/gdrive/MyDrive/Runs")
    print(results)

    # Evaluate the model's performance on the validation set
    results = model.val()
    print(results)

    # Export the model to ONNX format
    success = model.export(format="onnx")

    print(success)

if __name__ == '__main__':
    train()