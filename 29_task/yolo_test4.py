import cv2

from ultralytics import solutions

#cap = cv2.VideoCapture("C:\projects\yolo\app\.venv\29_road_video_2.mp4")
cap = cv2.VideoCapture("C:\\projects\\yolo\\app\\.venv\\29_road_video_2.mp4")
assert cap.isOpened(), "Error reading video file"

region_points = [(20, 400), (1080, 400)]                                      # line counting
#region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize object counter object
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
    # tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
#video_writer.release()
#cv2.destroyAllWindows()  # destroy all opened windows

cap.release()
video_writer.release()

print("✅ Обработка завершена. Нажмите любую клавишу чтобы закрыть окно...")

# Ожидание нажатия любой клавиши перед закрытием окон
cv2.waitKey(0)
cv2.destroyAllWindows()