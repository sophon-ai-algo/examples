#!/bin/bash
source model_info.sh

./download_from_nas.sh http://219.142.246.77:65000/sharing/SEZwCtk1M
mv ctdet_coco_dlav0_1x.pth ../build/
echo "[Success] ctdet_coco_dlav0_1x.pth has been downloaded in ../build/"

#url="https://docs.google.com/uc"
#method_name="export=download"
#file_id="18yBxWOlhTo32_swSug_HM4q3BeWgxp_N"
#
#if [ ! -f ${pth_model_name} ]; then
#   echo "Downloading ${pth_model_name}..."
#    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${file_id}'' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${file_id}" -O "${pth_model_name}" && rm -rf /tmp/cookies.txt
#   echo "Done!"
#else
#   echo "File already exists! Skip downloading procedure ..."
#fi
