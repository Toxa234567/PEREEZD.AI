import torch
import cv2
import os
from pathlib import Path
from utils.datasets import create_dataloader
from utils.general import check_dataset
from models.experimental import attempt_load

# Функция для разделения видео на кадры
def video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Сохраняем кадр как изображение
        cv2.imwrite(f'{output_dir}/frame_{frame_count}.jpg', frame)
        frame_count += 1

    cap.release()
    print(f"Saved {frame_count} frames to {output_dir}")

# Обучение модели
def train_model(data_yaml, video_path=None, epochs=50, batch_size=16, img_size=416):
    # Если есть видео, разделяем его на кадры и сохраняем в папку для обучения
    if video_path:
        video_to_frames(video_path, "images/train")  # Видео для обучения

    # Загружаем модель YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Используем предобученную модель YOLOv5

    # Создаём датасет и загрузчик
    train_loader, _ = create_dataloader("images/train", imgsz=img_size, batch_size=batch_size, augment=True, rect=True)
    val_loader = create_dataloader("images/val", imgsz=img_size, batch_size=batch_size, rect=True)

    # Настройка оптимизатора
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, targets, _, _ in train_loader:
            optimizer.zero_grad()

            # Прогон через модель
            loss, _ = model(imgs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    # Сохраняем обученную модель
    model.save(Path("runs/train/exp/weights/best.pt"))
    print("Model trained and saved!")

# Путь к видео для обучения
video_path = "images/video/video1.mp4"  # Укажи путь к видео

# Обучение модели на кадрах из видео
train_model('data.yaml', video_path=video_path)
