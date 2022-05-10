#!/bin/bash
split="===============================================================================================\n"
echo -e $split
command="grep"
   
# judge platform, on macOS we need use ggrep command
case "$OSTYPE" in
  solaris*) echo -e " OSType: SOLARIS! Aborting.\n"; exit 1 ;;
  darwin*)  echo -e " OSType: MACOSX!\n"; command="ggrep" ;;
  linux*)   echo -e " OSType: LINUX!\n" ;;
  bsd*)     echo -e " OSType: BSD! Aborting.\n"; exit 1 ;;
  msys*)    echo -e " OSType: WINDOWS! Aborting.\n"; exit 1 ;;
  *)        echo -e " OSType: unknown: $OSTYPE\n"; exit 1 ;;
esac
   
# judge if grep/ggrep exist
type $command >/dev/null 2>&1 || { echo >&2 "Using brew to install GUN grep first.  Aborting."; exit 1; }
   
if [ $# -eq 0 ];then
  echo -e "Usage: $0 sharing_file_path [save_path].\n"
  exit -1
fi
 
file_url=$1
 
host_prefix="https://disk.sophgo.vip" 
content=`curl -i --connect-timeout 3 $host_prefix`
if [ -z $content ];then
  host_prefix="http://219.142.246.77:65000"
fi
web_prefix=$host_prefix"/fsdownload/"
 
condition=`echo $file_url | cut -d "/" -f 4`
 
# single file
if [ $condition == "sharing" ];then
  echo -e "download single file"
  id=`echo $file_url | cut -d "/" -f 5`
  sid=`curl -i $file_url | $command  -Po '(?<=sid=).*(?=;path)'`
  v=`curl -i $file_url | $command -Po '(?<=none"&v=).*(?=">)'`
  file_name=`curl -b "sharing_sid=${sid}" -i ${host_prefix}"/sharing/webapi/entry.cgi?api=SYNO.Core.Sharing.Session&version=1&method=get&sharing_id=%22${id}%22&sharing_status=%22none%22&v=${v}" | $command -Po '(?<="filename" : ").*(?=")'`
 
  if [ $# -eq 2 ];then
    save_path=$2
  else
    save_path=$file_name
  fi
 
  echo -e "\ndownload with sid=$sid\n"
  curl -o $save_path -b "sharing_sid=${sid}" "${web_prefix}${id}/${file_name}"
  echo -e "\nDone! Saved to $save_path.\n"
  echo -e $split
fi
# files with zip
if [ $condition == "fsdownload" ];then
  echo -e "download files with zip"
  id=`echo $file_url | cut -d "/" -f 5`
  sid=`curl -i "${host_prefix}/sharing/${id}" | $command  -Po '(?<=sid=).*(?=;path)'`
  file_name=`echo $file_url | cut -d "/" -f 6`
  zip_name=`echo $file_url | cut -d "/" -f 6`".zip"
  if [ $# -eq 2 ];then
    save_path=$2
  else
    save_path=$zip_name
  fi
  echo -e "\ndownload with sid=$sid\n"
  post_data="api=SYNO.FolderSharing.Download&method=download&version=2&mode=download&stdhtml=false&dlname=%22${zip_name}%22&path=%5B%22%2F${file_name}%22%5D&_sharing_id=%22${id}%22&codepage=chs"
  curl -o $save_path -H "Content-Type: application/x-www-form-urlencoded" -b "sharing_sid=${sid}" -X POST -d $post_data "${web_prefix}webapi/file_download.cgi/${zip_name}"
  echo -e "\nDone! Saved to $save_path.\n"
  echo -e $split
fi