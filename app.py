from flask import Flask, render_template, request
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']

        if image.filename == '':
            return "파일을 선택하세요."

        if image:
            filename = secure_filename(image.filename)
            image.save(filename)
            return process_image(filename)

@app.route('/process/<image_path>')
def process_image(image_path):
    # YOLO 설정 파일과 가중치 파일
    yolo_config = 'yolov4.cfg'
    yolo_weights = 'yolov4.weights'

    # 클래스 이름 파일
    yolo_names = 'coco.names'

    # YOLO 모델 불러오기
    net = cv2.dnn.readNet(yolo_config, yolo_weights)

    # 클래스 이름 불러오기
    classes = []
    with open(yolo_names, 'r') as f:
        classes = f.read().strip().split('\n')

    # 이미지 불러오기
    image = cv2.imread(image_path)

    # 이미지 크기 설정
    height, width, channels = image.shape

    # YOLO 입력 이미지로 변환
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # YOLO 감지 수행
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # 경계 상자, 클래스 ID 및 신뢰도 추출
    conf_threshold = 0.5
    nms_threshold = 0.4

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # 비최대 억제 (Non-Maximum Suppression) 수행
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 결과 이미지에 객체와 경계 상자 그리기
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 결과 이미지 저장
    output_path = 'static/output_image.jpg'  # 이미지를 'static' 폴더에 저장
    cv2.imwrite(output_path, image)

    return render_template('result.html', output_path=output_path)

if __name__ == '__main__':
    app.run(debug=True)