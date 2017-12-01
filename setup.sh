#!/bin/bash

for i in `seq 1 3`; do
    cd Proiect$i;
    ./setup.sh;
    cd ..;
done
