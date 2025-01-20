#!/bin/bash

# 文件夹 ID
FOLDER_ID="1DbzEcJxkBYVuOb2MACML14SWNZveAVuH"

# 使用 gdown 下载文件夹
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O model_20241223/

echo "Model saved!"
