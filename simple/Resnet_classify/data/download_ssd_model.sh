#-----------------------------------------
# Download Script
#   - from internal website
#-----------------------------------------
#!/bin/bash
url="http://10.30.34.184:8192/imagenet_2k/raw/resnet50.int8.bmodel"
cur_dir=${PWD##*/}

echo "Downloading ${file_name}..."
wget -c "$url"
echo "Done!"
