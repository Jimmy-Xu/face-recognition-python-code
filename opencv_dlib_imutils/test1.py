from imutils.video import FileVideoStream
from imutils.video import VideoStream
import imutils
import dlib
import cv2
import sys

"""
pip3 install imutils
"""

def _help():
    print("Usage:")
    print("     python landmark_detect.py")
    print("     python landmark_detect.py <path of a video>")
    print("For example:")
    print("     python landmark_detect.py video/lee.mp4")
    print("If the path of a video is not provided, the camera will be used as the input.Press q to quit.")


def landmark_detection(vs, file_stream):
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("..\\simple_faceswap\\models\\shape_predictor_68_face_landmarks.dat")

    print("[INFO] starting video stream thread...")
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if file_stream and not vs.more():
            break

        frame = vs.read()
        if frame is not None:
            frame = imutils.resize(frame, width=450) #调节显示的像素大小 450我的超薄本可以很流畅 实时用dlib识别人脸不卡顿 电脑好的调高画质 效果不错
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                # shape = face_utils.shape_to_np(shape)
                chang = []
                kuan = []
                for idx, pt in enumerate(shape.parts()):
                    pt_pos = (pt.x, pt.y)
                    chang.append(pt.x)
                    kuan.append(pt.y)
                    #cv2.circle(frame, pt_pos, 1, (0, 0, 255), 1)
                    #font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(frame, str(idx + 1), pt_pos, font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
                x1 = max(chang)
                x2 = min(chang)
                y1 = max(kuan)
                y2 = min(kuan)
                cv2.rectangle(frame, (x2, y2), (x1, y1), (0, 255, 0), 2)
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()


if len(sys.argv) > 2 or "-h" in sys.argv or "--help" in sys.argv:
    _help()
elif len(sys.argv) == 2:
    vs = FileVideoStream(sys.argv[1]).start()
    file_stream = True
    landmark_detection(vs, file_stream)
else:
    vs = VideoStream(src=0).start() # 1代表外置摄像头 0代表笔记本自带摄像头
    file_stream = False
    landmark_detection(vs, file_stream)