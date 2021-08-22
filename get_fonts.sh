#! /usr/bin/bash
mkdir fonts
wget -O noto_sans_jp.zip https://fonts.google.com/download?family=Noto%20Sans%20JP
unzip noto_sans_jp.zip -d fonts -x *OFL.txt
wget -O noto_serif_jp.zip https://fonts.google.com/download?family=Noto%20Serif%20JP
unzip noto_serif_jp.zip -d fonts -x *OFL.txt
