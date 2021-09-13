import face_recognition
import cv2


# 方法-图中某个区域磨皮美颜
def do_beautify(img, top, right, bottom, left):
    # 双边滤波-磨皮美颜
    beautify_img = cv2.bilateralFilter(img[top: bottom, left:right], 10, 20, 20)
    # 修改原图
    height, width, channels = beautify_img.shape
    for i in range(0, height):
        for j in range(0, width):
            img[top + i, left + j] = beautify_img[i, j]


# 开始处理
video_caputre = cv2.VideoCapture(0)
while True:
    ret, frame = video_caputre.read()
    if ret:
        # 转换 BGR(opencv使用) 为 RGB(face_recognition使用)
        rgb_img = frame[:, :, ::-1]
        # 识别图片中的脸部
        face_locations = face_recognition.face_locations(rgb_img)
        # 遍历人脸，进行识别
        for (top, right, bottom, left) in face_locations:
            # 双边滤波-磨皮美颜
            do_beautify(frame, top, right, bottom, left)
            # 画框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

