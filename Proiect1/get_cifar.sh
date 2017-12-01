#!/bin/bash

echo "Downloading cifar"
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar.tar.gz
tar xf cifar.tar.gz
rm cifar.tar.gz
