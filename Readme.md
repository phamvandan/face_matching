Guide to run
---
Step 1: Install requirements.txt  
Step 2: Run these commands.
```
    mkdir mta_test
    git clone https://github.com/deepinsight/insightface.git
    git clone https://github.com/phamvandan/face_matching.git
    cd face_matching/
    python3 face_detector.py -f path/to/image/folder -sf path/to/save/folder
```
Results
---
* The format of log file as follow:  
* result.txt: 
```
Total images:
// total images processed
img_height:
// use for detection
Can not detect: 
// number of images can not be detected
Rotate: 
// number of rotations 
Avg rotate time: 
// avg time of rotatation
Total times:
// total times for process
Avg times:   
// avg times per image
```
* details.txt:
```
    file_name (img_w, img_h) number_of_face [face_x, face_y, face_w, face_h confidence] number_of_rotation time
```
