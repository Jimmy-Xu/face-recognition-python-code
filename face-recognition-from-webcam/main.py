#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/3 15:52
# @Author  : He Hangjiang
# @Site    :
# @File    : 摄像头实时识别.py
# @Software: PyCharm

import face_recognition
import cv2

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



video_capture = cv2.VideoCapture(0)

# 本地图像
hhj_image = face_recognition.load_image_file("xjimmy.jpg")
hhj_face_encoding = face_recognition.face_encodings(hhj_image)[0]

#
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 读取摄像头画面
    ret, frame = video_capture.read()

    # 改变摄像头图像的大小，图像小，所做的计算就少
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 默认为unknown
            match = face_recognition.compare_faces([hhj_face_encoding], face_encoding)
            name = "Unknown"

            if match[0]:
                name = u"呆鱼"

            face_names.append(name)

    process_this_frame = not process_this_frame

    # 将捕捉到的人脸显示出来
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #加上标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        # 英文
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # 中文
        frame = cv2AddChineseText(frame, name, (left + 10, bottom), (255, 255, 255), 30)

    # Display
    cv2.imshow('Video', frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




video_capture.release()
cv2.destroyAllWindows()