#!/bin/bash
sudo apt update
sudo apt install curl
script_dir=$(dirname $(readlink -f "$0"))
root_dir=$script_dir/..
data_dir=$root_dir/data
if [ ! -d "$data_dir" ]; then
  echo "create data dir: $data_dir"
  mkdir -p $data_dir
fi

./download_from_nas.sh http://219.142.246.77:65000/sharing/r1ENDQWz3
./download_from_nas.sh http://219.142.246.77:65000/sharing/X9kDZRcKJ
./download_from_nas.sh http://219.142.246.77:65000/sharing/0AZ6cglFe
./download_from_nas.sh http://219.142.246.77:65000/sharing/vv7sw4FL5
./download_from_nas.sh http://219.142.246.77:65000/sharing/2LqvHTw2I
./download_from_nas.sh http://219.142.246.77:65000/sharing/QCOAQGj3i

./download_from_nas.sh http://219.142.246.77:65000/sharing/iLmuKYi1F

pushd $data_dir

image_dir=$data_dir/images
if [ ! -d "$image_dir" ]; then
  echo "create image dir: $image_dir"
  mkdir -p $image_dir
fi
pushd $image_dir
mv $script_dir/000000162415.jpg ./
mv $script_dir/000000250758.jpg ./
mv $script_dir/000000404484.jpg ./
mv $script_dir/000000404568.jpg ./
mv $script_dir/n02412080_66.JPEG ./
mv $script_dir/n07697537_55793.JPEG ./
popd

video_dir=$data_dir/videos
if [ ! -d "$video_dir" ]; then
  echo "create video dir: $video_dir"
  mkdir -p $video_dir
fi
pushd $video_dir
mv $script_dir/road.mp4 ./
popd

popd

