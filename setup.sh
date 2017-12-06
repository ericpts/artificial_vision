#!/bin/bash

git submodule update --init

for i in `seq 1 3`; do
    cd Proiect$i;
    ./setup.sh;
    cd ..;
done

sudo pip3 install -r requirements.txt
pushd lib_google_img
sudo pip3 install -r requirements-dev.txt
popd
