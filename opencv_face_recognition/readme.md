# install dependency
```
pip3 install face_recognition dlib  opencv-python cmake numpy keras Pillow keras
```

# prepare

- add each face to face_images
- add emotion.jpg to img/emotion/
- clone https://github.com/opencv/opencv to opencv dir
- add simsun.ttc to fonts dir

# usage

```
python3 1_from_image.py
python3 2_from_webcam.py
python3 3_people.py
python3 4_people_faster.py
python3 5_mosaic.py
python3 6_beautify.py
python3 7_emotion.py
```