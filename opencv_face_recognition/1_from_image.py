import face_recognition
import cv2
import os

# 读取图片
img = cv2.imread(".\\img\\emotion\\emotion.jpg")
# 转换 BGR(opencv使用) 为 RGB(face_recognition使用)
rgb_img = img[:, :, ::-1]

# 识别图片中的脸部（可能存在多个脸）
face_locations = face_recognition.face_locations(rgb_img)

# 遍历人脸位置信息
for top, right, bottom, left in face_locations:
    # 对人脸画框
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

# 后面显示还是用img显示（BGR），不要用rgb_img显示，否则你会得到意向不到的“阿凡达”特效，哈哈
# 展示，按q退出
cv2.imshow("img", img)
if cv2.waitKey() == ord("q"):
    cv2.destroyAllWindows()
