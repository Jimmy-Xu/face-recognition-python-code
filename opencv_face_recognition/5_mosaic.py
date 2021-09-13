import face_recognition
import cv2


# 方法-为图片某个区域打上马赛克
def do_mosaic(img, top, right, bottom, left, step):
    for i in range(top, bottom, step):
            for j in range(left, right, step):
                for y in range(0, step):
                    for x in range(0, step):
                        img[i + y, j + x] = img[i, j]


# 开始处理
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if ret:
        # 转换 BGR(opencv使用) 为 RGB(face_recognition使用)
        rgb_img = frame[:, :, ::-1]
        # 识别图片中的脸部
        face_locations = face_recognition.face_locations(rgb_img)
        # 遍历人脸，进行识别
        for (top, right, bottom, left) in face_locations:
            # 打上马赛克
            do_mosaic(frame, top, right, bottom, left, 8)
            # 画框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
