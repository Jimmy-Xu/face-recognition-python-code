import face_recognition
import cv2
import os

# for cv2AddChineseText
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



"""
准备一个文件夹（或者数据库），放入你的人脸图片(一张图片一个脸)，例如
./face_images/xxx.jpg
./face_images/yyy.jpg
"""


"""
读取已经准备好的人脸图片信息
"""
# 人姓名
person_names = []
# 人面部信息
person_face_encodings = []

img_dir = "./face_images/"
files = os.listdir(img_dir)
for file in files:
    if file.endswith("jpg") or file.endswith("png"):
        # 去除文件后缀类型
        name, _ = os.path.splitext(file)
        person_names.append(name)
        # 拼接图片完整路径
        img_path = os.path.join(img_dir, file)
        # 解析出已有图片的脸部信息
        img_file = face_recognition.load_image_file(img_path)
        face_endocing = face_recognition.face_encodings(img_file)[0]
        person_face_encodings.append(face_endocing)

"""
捕获摄像头，并对比已知的人脸信息
"""
# 捕获摄像头
# 优化点
# 1. 缩小原图，再进行识别，提高了识别的速度
# 2. 交替，每隔一帧，进行一次识别，降低了识别的频次

video_capture = cv2.VideoCapture(0)

# 初始化变量
face_locations = []  # 脸部位置列表
face_encodings = []  # 脸部编码列表
face_names = []  # 脸的人名列表
process_this_frame = True  # 是否识别当前帧

while True:
    # 读取帧
    ret, frame = video_capture.read()
    # 读取失败，就退出
    if not ret:
        break

    # 缩放当前帧为1/4，以提高后面的识别速度
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 转换BGR为RGB，方便face_recognition使用
    rgb_small_frame = small_frame[:, :, ::-1]

    # 识别处理
    if process_this_frame:
        # 获取当前帧的人脸位置、编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 人名、近似度的默认值
            name = "Unknown"
            similarity = 0.0

            # 计算当前脸与已知的脸的欧氏距离，越小越相近
            face_distances = face_recognition.face_distance(person_face_encodings, face_encoding)
            # 将当前人脸编码与已知的所有人脸编码进行比对，确定是否匹配
            matches = face_recognition.compare_faces(person_face_encodings, face_encoding)

            # 获取第一个匹配的人名
            for index, match in enumerate(matches):
                if match:
                    name = person_names[index]
                    similarity = face_distances[index]
                    break

            face_names.append((name, similarity))
    # 交替 True, False, True, ...
    # 保证每隔一帧进行一次识别，而不是每一帧都识别
    process_this_frame = not process_this_frame

    # 显示面部的框、人名
    for (top, right, bottom, left), (name, similarity) in zip(face_locations, face_names):
        # 由于前面是从小图的中识别的脸，因此此处要扩大回原来的比例
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 画脸的框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 画文字
        # 英文
        # cv2.putText(frame, "%s, %.2f" % (name, similarity), (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0),1)
        # 中文
        frame = cv2AddChineseText(frame, "%s, %.2f" % (name, similarity), (left + 10, bottom), (255, 255, 255), 30)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
