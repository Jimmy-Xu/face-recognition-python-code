import face_recognition as fr
import cv2
import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont
import sys

"""
pip3 install cmake
pip3 install dlib-19.22.99-cp39-cp39-win_amd64.whl  #下载： https://github.com/Jimmy-Xu/dlib_compiled
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip3 install face_recognition 

pip3 install Pillow
"""


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


def main(filename):
    path = "train/"

    known_names = []
    known_name_encodings = []

    images = os.listdir(path)
    for _ in images:
        image = fr.load_image_file(path + _)
        image_path = path + _
        encoding = fr.face_encodings(image)[0]

        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

        # 化妆
        face_landmarks_list = fr.face_landmarks(image)
        for face_landmarks in face_landmarks_list:
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image, 'RGBA')

            # Make the eyebrows into a nightmare
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

            # Gloss the lips
            d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
            d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

            # Sparkle the eyes
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

            # Apply some eyeliner
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
            pil_image.show()

    print(known_names)

    test_image = os.path.join(os.getcwd (), "test", filename)

    if not os.path.exists( test_image ):
        print("test image " + test_image + " not found")
        exit()

    image = cv2.imread(test_image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        face_distances = fr.face_distance(known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # 英文
        # cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # 中文
        cv2.rectangle(image, (left, bottom), (right, bottom + 30), (0, 0, 255), cv2.FILLED)
        image = cv2AddChineseText(image, name, (left + 10, bottom), (255, 255, 255), 30)

    cv2.imshow("Result", image)
    cv2.imwrite("output.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python main.py <image_file>")
        exit()
    main(sys.argv[1])
