#!/bin/bash

yapf --parallel \
    --exclude './lib_google_img/*' \
    --recursive \
    --in-place \
    .
