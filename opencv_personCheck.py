import numpy as np
import cv2
import time
import datetime
from PIL import ImageFont, ImageDraw, Image


# 카메라 영상을 받아온 객체 선언 및 설정
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#fonts파일 안에 글꼴 불러옴
font = ImageFont.truetype('fonts/SCDream6.otf', 20)

#객체 이미지 저장하는 변수 초기화
h, w = None, None

#coco.names 파일을 열어서 각 줄을 라벨로 80개 저장함
with open('coco.names') as f:
    labels = [line.strip() for line in f]

#yolov3의 구성 파일이랑 딥러닝 된 가중치를 사용해 네트워크를 초기화함
network = cv2.dnn.readNetFromDarknet('yolov3.cfg',
                                     'yolov3.weights')
layers_names_all = network.getLayerNames()
#네트워크의 모든 이름을 가져옴

#네트웤의 출력 이름을 가져옴
layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

#객체 인식에 사용할 최소 확률값 설정
probability_minimum = 0.5

threshold = 0.3

#객체 박스 색깔 무작위
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

#무한루프
while True:
    #camera = cv2.VideoCapture(0) 카메라에서 프레임을 읽어옴.
    _, frame = camera.read()
    #프레임 색상을 회색으로 바꿈 이걸 해야 인식이 더 잘됨.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #객체 높이와 너비를 확인하고 값에 저장
    if w is None or h is None:
        h, w = frame.shape[:2]

    #신경망이 처리할 수 있는 형태로 변환 이미지 크기 조정 및 색상을 정규화
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=True)

    #신경망의 입력으로 설정
    network.setInput(blob)
    #신경망을 통해 전방향 계산을 수행하고 결과를 가져옴.
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    #프레임을 처리하는데 걸린 시간을 터미널에 출력
    print('Current frame took {:.5f} seconds'.format(end - start))

    # 객체 인식 상자, 인식 %, 객체명를 저장할 곳 초기화
    bounding_boxes = []
    confidences = []
    classIDs = []

    
    for result in output_from_network:
        for detected_objects in result: #밑에 값을 반복
            scores = detected_objects[5:] #객체 계산
            class_current = np.argmax(scores) #가장 높은 점수를 가진 객채 선택
            confidence_current = scores[class_current] # 선택된 객체 %표시

            #class_current == 0은 person만 확인하는 코드#
            if class_current == 0 and confidence_current > probability_minimum:
                #객체 인식 상자 계산 #detected_objects는 상자의 중심 좌표와 너비 높이를 나타내는 상대값
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                #객체 상자 왼쪽 상단 좌표 계산
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                
                #계산된 상자를 리스트에 추가
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                #객체 %지 이름을 리스트에 저장
                confidences.append(float(confidence_current))
                classIDs.append(class_current)

    #인식이 낮거나 겹치는 객체 인식 상자 제거
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
    
    #사람 인식 변수 초기화
    object_count = 0

    #감지된 객체가 있다면 실행됨
    if len(results) > 0:
        for i in results.flatten():
            # 객체 이름이 person이면 1씩 증가
            if labels[classIDs[i]] == 'person':
                object_count += 1
    #사람의 수를 문자열로 변환.
    text_count = 'Person Count: {}'.format(object_count)

    #사람의 수를 영상에 표시함.
    cv2.putText(frame, text_count, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


    if len(results) > 0:
        for i in results.flatten():
            #객체 인식 정보 상자 정보를 가져옴
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            #객체와 객체 이름 색을 가져옴
            colour_box_current = colours[classIDs[i]].tolist()
            
            #객체 인식상자를 화면에 나타냄
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            #객체 이름과 %지를 문자열로 변환
            text_box_current = '{}: {:.4f}'.format(labels[int(classIDs[i])],
                                                   confidences[i])
            #화면에 표시한다.
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    # 현재시각을 불러와 문자열로 저장함
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    #글자 뒤에 배경을 넣음 pt1과 pt2는 사각형의 시작점과 끝점 thickness 선굵기#
    cv2.rectangle(img=frame, pt1=(10, 15), pt2=(405, 35), color=(0,0,0), thickness=-1)
    
    #영상에 글자를 더해주는 역할을 함
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    # xy는 텍스트 시작위치임, 
    draw.text(xy=(10, 15),  text="송이최이 식당 혼잡도 "+nowDatetime, font=font, fill=(255, 255, 255))
    frame = np.array(frame)
    
    #현재 시간을 표시하는 영상 출력
    cv2.imshow("original", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #q입력시 무한루프 종료
        break

# camera.release() #캡처 객체 없앰
# cv2.destroyAllWindows() #모든 영상 창 닫음