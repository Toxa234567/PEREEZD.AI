import torch
import cv2
import os

# Функция для предсказаний на изображениях
def predict_on_image(model, image_path):
    img = cv2.imread(image_path)  # Загружаем изображение
    results = model(img)  # Прогон через модель
    results.show()  # Покажет окно с изображением и рамками вокруг обнаруженных объектов
    detections = results.pandas().xywh  # Выводим DataFrame с результатами
    print(detections)

# Функция для предсказаний на видео
def predict_on_video(model, video_path):
    cap = cv2.VideoCapture(video_path)  # Загружаем видео
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Прогон через модель
        results = model(frame)

        # Отображение результатов
        results.show()  # Покажет окно с изображением и рамками вокруг обнаруженных объектов

        # Выводим DataFrame с результатами
        detections = results.pandas().xywh
        print(detections)

    cap.release()
    cv2.destroyAllWindows()

# Загружаем обученную модель
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')  # Путь к обученной модели

# Пример для предсказания на изображении
image_path = "your_image.jpg"  # Укажи путь к изображению
predict_on_image(model, image_path)

# Пример для предсказания на видео
video_path = "images/video/video1.mp4"  # Укажи путь к видео
predict_on_video(model, video_path)
