echo -e "Downloading glove (in use by the evaluators)"
gdown --proxy 127.0.0.1:1087 --fuzzy https://drive.google.com/file/d/1bCeS6Sh_mLVTebxIgiUHgdPrroW06mb6/view?usp=sharing
rm -rf glove

unzip glove.zip
echo -e "Cleaning\n"
rm glove.zip

echo -e "Downloading done!"