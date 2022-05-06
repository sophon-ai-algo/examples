apt update
apt install -y curl unzip
curl -L -o coco128.zip https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
unzip coco128.zip
rm coco128.zip