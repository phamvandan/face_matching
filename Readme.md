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
img_height: // use for detection
Can not detect: 
Rotate: // number of rotations 
Avg rotate time: // time of rotate
Total times:
Avg times:   
```
* face_size.txt: size of cropped faces
* origin_image_size.txt: size of original images