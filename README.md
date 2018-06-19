# car_count_yolo
COE POC of real time object detection use yolov3 model 

Grab the pretrained weights of yolo3 from 
```https://pjreddie.com/media/files/yolov3.weights.```

To dowload the video sample: 
```https://drive.google.com/file/d/192nGFPszo8VM3Fnlp7GT1zWbiMLCo5Cf/view```

### real time webcam detection:
```python -m yolo.camera_detection_array -w yolov3.weights```

### car count video detection:
Detection <br>
```python -m yolo.car_count_zone_video -i image -w yolov3.weights -v video_sample.mp4```<br>
or<br>
```python -m yolo.car_count_zone_video -i image -w yolov3.weights -u rtsp://count:mdp@xx.xx.xx.xx:xxxx/live/ch0```<br>
Run flask server <br>
```python yolo/flask_server/Ec2_pc.py```<br>
Put this link in your chrome explorer <br>
```http://127.0.0.1:5000/video_feed```<br>
