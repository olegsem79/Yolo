import ultralytics
import cv2  
from ultralytics import solutions

ultralytics.checks()

import torch
import ultralytics

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ CUDA не доступна. Проверьте установку драйверов и PyTorch с GPU поддержкой")

import subprocess
import sys

def check_gpu():
    """Проверка наличия NVIDIA GPU в системе"""
    try:
        # Для Windows
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU обнаружена")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA GPU не обнаружена или драйверы не установлены")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi не найден. Установите драйверы NVIDIA")
        return False

check_gpu()

cap = cv2.VideoCapture(0)  # Веб-камера
# Устанавливаем высокое разрешение ДО assert
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Ширина 1920px
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Высота 1080px
cap.set(cv2.CAP_PROP_FPS, 30)             # FPS 30

assert cap.isOpened(), "Error reading video file"

# Получаем УСТАНОВЛЕННЫЕ параметры
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📹 Разрешение: {w}x{h}, FPS: {fps}")

# Video writer
video_writer = cv2.VideoWriter("counting.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

# Define region points
region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolo11n.pt",  # Используйте существующую модель
)

print("🎥 Обработка видео начата. Нажмите 'q' для выхода, 'p' для паузы.")

# Process video
paused = False
while cap.isOpened():
    if not paused:
        success, im0 = cap.read()
        if not success:
            print("Видео завершено или ошибка чтения.")
            break
        
        results = counter(im0)
        video_writer.write(results.plot_im)
    
    # Обработка клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Выход
        print("🚪 Выход по нажатию 'q'")
        break
    elif key == ord('p'):  # Пауза/продолжение
        paused = not paused
        print("⏸️ Пауза" if paused else "▶️ Продолжение")
    elif key == ord('s'):  # Сделать скриншот
        cv2.imwrite(f"screenshot_{cv2.getTickCount()}.jpg", results.plot_im)
        print("📸 Скриншот сохранен")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("✅ Программа завершена")