import cv2
import time

colors = [(0, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

while True:
    _, frame = cap.read()

    start = time.time()

    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    end = time.time()

    for(classid, score, box) in zip(classes, scores, boxes):


        color = colors[int(classid) % len(colors)]
        label = f"{class_names[classid]} : {round(score*100, 2)}"

        cv2.rectangle(frame, box, color, 2)

        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)

    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("detections", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()