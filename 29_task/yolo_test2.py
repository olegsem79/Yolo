import ultralytics
import cv2  
from ultralytics import solutions
import torch
import os
from datetime import datetime

ultralytics.checks()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Создаем папку для скриншотов
screenshots_dir = "detection_screenshots"
os.makedirs(screenshots_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

assert cap.isOpened(), "Error reading video file"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📹 Разрешение: {w}x{h}, FPS: {fps}")

video_writer = cv2.VideoWriter("counting.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

# Define region points
region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolov8n.pt",  # Исправлено на yolov8n.pt
)

print("🎥 Обработка видео начата. Нажмите 'q' для выхода.")

# Переменные для управления скриншотами
last_screenshot_time = 0
screenshot_cooldown = 2  # Минимальная пауза между скриншотами (секунды)
screenshot_count = 0

def save_detection_screenshot(frame, detections_count):
    """Сохраняет скриншот с обнаружениями"""
    global screenshot_count, last_screenshot_time
    
    current_time = datetime.now()
    
    # Проверяем cooldown
    if cv2.getTickCount() / cv2.getTickFrequency() - last_screenshot_time < screenshot_cooldown:
        return
    
    screenshot_count += 1
    timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{screenshots_dir}/detection_{timestamp}_count_{detections_count}.jpg"
    
    # Добавляем информацию на скриншот
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, f"Detections: {detections_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, current_time.strftime("%Y-%m-%d %H:%M:%S"), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(filename, annotated_frame)
    print(f"📸 Авто-скриншот: {filename}")
    
    last_screenshot_time = cv2.getTickCount() / cv2.getTickFrequency()

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Видео завершено или ошибка чтения.")
        break
    
    results = counter(im0)
    video_writer.write(results.plot_im)
    
    # Проверяем, есть ли обнаружения в зоне
    if hasattr(counter, 'in_count') and counter.in_count > 0:
        save_detection_screenshot(im0, counter.in_count)
    
    # Обработка клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🚪 Выход по нажатию 'q'")
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"✅ Программа завершена. Сохранено скриншотов: {screenshot_count}")