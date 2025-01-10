rm -rf checkpoints
mkdir checkpoints
cd checkpoints
echo -e "Downloading extractors"
#gdown --proxy 127.0.0.1:1087 --fuzzy https://drive.google.com/file/d/1o7RTDQcToJjTm9_mNWTyzvZvjTWpZfug/view
#gdown --proxy 127.0.0.1:1087 --fuzzy https://drive.google.com/file/d/1KNU8CsMAnxFrwopKBBkC8jEULGLPBHQp/view

gdown --proxy 127.0.0.1:1087 --fuzzy https://drive.google.com/file/d/1IgrFCnxeg4olBtURUHimzS03ZI0df_6W/view
gdown --proxy 127.0.0.1:1087 --fuzzy https://drive.google.com/file/d/12liZW5iyvoybXD8eOw4VanTgsMtynCuU/view


unzip t2m.zip
unzip kit.zip

echo -e "Cleaning\n"
rm t2m.zip
rm kit.zip
echo -e "Downloading done!"