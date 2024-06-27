import ultralytics
ultralytics.checks()
!yolo predict model=yolov8n.pt source='https://ultralytics.com/images/zidane.jpg'