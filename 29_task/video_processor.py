# video_processor.py
# video_processor.py
import os
from pathlib import Path
from supervision import (
    VideoInfo, 
    VideoSink, 
    get_video_frames_generator,
    Detections,
    LineZone,
    LineZoneAnnotator,
    TraceAnnotator,
    Point,
    Position,
    Color,
    ByteTrack  # Импортируем ByteTrack из supervision!
)
from ultralytics import YOLO


def process_video_with_lines(
    source_video_path: str | os.PathLike,
    target_video_path: str | os.PathLike,
    model: YOLO,
    tracker: ByteTrack,
    line_top_start: Point = Point(0, 180),
    line_top_end: Point = Point(768, 180),
    line_bottom_start: Point = Point(0, 350),
    line_bottom_end: Point = Point(768, 350),
    trace_length: int = 50
):
    """
    Обрабатывает видео с детекцией, трекингом и подсчетом пересечений линий
    
    Args:
        source_video_path: Путь к исходному видео
        target_video_path: Путь для сохранения результата
        model: YOLO модель для детекции
        tracker: Трекер для отслеживания объектов
        line_top_start: Начальная точка верхней линии
        line_top_end: Конечная точка верхней линии  
        line_bottom_start: Начальная точка нижней линии
        line_bottom_end: Конечная точка нижней линии
        trace_length: Длина трейсов объектов
    """
    
    # Инициализация линий
    line_top = LineZone(
        line_top_start, 
        line_top_end, 
        [Position.TOP_CENTER, Position.TOP_LEFT, Position.TOP_RIGHT]
    )
    line_bottom = LineZone(
        line_bottom_start, 
        line_bottom_end, 
        [Position.BOTTOM_CENTER, Position.BOTTOM_LEFT, Position.BOTTOM_RIGHT]
    )
    
    # Инициализация аннотаторов
    line_top_annotator = LineZoneAnnotator(color=Color.RED, display_out_count=False)
    line_bottom_annotator = LineZoneAnnotator(color=Color.RED, display_in_count=False)
    trace_annotator = TraceAnnotator(trace_length=trace_length)
    
    # Обработка видео
    gen = get_video_frames_generator(source_video_path)
    video_info = VideoInfo.from_video_path(source_video_path)
    
    with VideoSink(target_video_path, video_info) as sink:
        for frame_num, frame in enumerate(gen):
            # Детекция
            yolo_detections = model(frame, verbose=False)[0]
            detections = Detections.from_ultralytics(yolo_detections)
            
            # Трекинг
            detections = tracker.update_with_detections(detections)
            
            # Аннотация
            annotated_frame = trace_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            
            # Обработка пересечений линий
            line_top.trigger(detections)
            line_bottom.trigger(detections)
            
            # Аннотация линий
            annotated_frame = line_top_annotator.annotate(annotated_frame.copy(), line_top)
            annotated_frame = line_bottom_annotator.annotate(annotated_frame.copy(), line_bottom)
            
            sink.write_frame(annotated_frame)
            
            # Прогресс (опционально)
            if frame_num % 30 == 0:
                print(f"Обработано кадров: {frame_num}")
    
    # Возвращаем статистику
    return {
        'in_count': line_top.in_count,
        'out_count': line_bottom.out_count,
        'total_detections': line_top.in_count + line_bottom.out_count
    }