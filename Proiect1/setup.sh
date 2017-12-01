#!/bin/bash

tar xf data.tar.gz

if [ -d cifar-10-batches-py ]; then
    echo "Found cifar in cifar-10-batches-py"
else
    echo "Did not find cifar. Trying to download it."
    ./get_cifar.sh
fi
