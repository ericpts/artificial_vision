#!/bin/bash

autopep8 --jobs $(nproc) --experimental  --max-line-length 100 --in-place --aggressive --aggressive --aggressive --aggressive --recursive . --exclude './lib_google_img/*'
