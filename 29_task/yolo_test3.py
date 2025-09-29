import ultralytics
import cv2  
import torch
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Создаем папку для скриншотов
screenshots_dir = "smart_detections"
os.makedirs(screenshots_dir, exist_ok=True)

# Загружаем модель YOLO
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Region of Interest (ROI)
roi_points = np.array([(20, 400), (1080, 400), (1080, 360), (20, 360)])

def is_in_roi(box, roi):
    """Проверяет находится ли bounding box в зоне интереса"""
    x1, y1, x2, y2 = box
    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    # Проверяем находится ли центр бокса в полигоне ROI
    return cv2.pointPolygonTest(roi, box_center, False) >= 0

def save_smart_screenshot(frame, detections_info):
    """Сохраняет умный скриншот с информацией об обнаружениях"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # Создаем аннотированный frame
    annotated_frame = frame.copy()
    
    # Рисуем ROI
    cv2.polylines(annotated_frame, [roi_points], True, (0, 255, 0), 2)
    cv2.putText(annotated_frame, "DETECTION ZONE", (roi_points[0][0], roi_points[0][1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Добавляем информацию об обнаружениях
    y_offset = 30
    for class_name, count in detections_info.items():
        text = f"{class_name}: {count}"
        cv2.putText(annotated_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    cv2.putText(annotated_frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Сохраняем
    filename = f"{screenshots_dir}/detection_{timestamp}.jpg"
    cv2.imwrite(filename, annotated_frame)
    print(f"📸 Умный скриншот: {filename}")
    return filename

# Основной цикл
last_detection_time = 0
cooldown = 2  # секунды
detection_count = 0

print("🎥 Умная детекция запущена. Скриншоты при обнаружении в зоне.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Детекция
    results = model(frame, verbose=False)
    
    # Анализируем обнаружения в ROI
    roi_detections = {}
    for result in results:
        for box in result.boxes:
            if is_in_roi(box.xyxy[0].cpu().numpy(), roi_points):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                roi_detections[class_name] = roi_detections.get(class_name, 0) + 1
    
    # Сохраняем скриншот если есть обнаружения в ROI
    if roi_detections and (datetime.now().timestamp() - last_detection_time > cooldown):
        save_smart_screenshot(frame, roi_detections)
        last_detection_time = datetime.now().timestamp()
        detection_count += 1
    
    # Визуализация
    annotated_frame = results[0].plot()
    cv2.polylines(annotated_frame, [roi_points], True, (0, 255, 0), 2)
    cv2.imshow('Smart Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Сохранено умных скриншотов: {detection_count}")