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
video_caputre = cv2.VideoCapture(0)

while True:
    ret, frame = video_caputre.read()
    if ret:
        # 转换 BGR(opencv使用) 为 RGB(face_recognition使用)
        rgb_img = frame[:, :, ::-1]
        # 识别图片中的脸部
        face_locations = face_recognition.face_locations(rgb_img)
        # 对识别出的面部区域编码
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        # 遍历人脸，进行识别
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 画框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # 将图中的面部与已有的面部进行对比，得到姓名
            matches = face_recognition.compare_faces(person_face_encodings, face_encoding)
            name = "Unknown"
            for index, match in enumerate(matches):
                if match:
                    name = person_names[index]
                    break
            # 写姓名
            # 英文
            # cv2.putText(frame, name, (left, top - 10), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
            # 中文
            frame = cv2AddChineseText(frame, name, (left + 10, bottom), (255, 255, 255), 30)

        cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_caputre.release()
cv2.destroyAllWindows()
