#!/usr/bin/env bash

pip freeze >> requirements.txt
git add .
git commit -m "updated requirements.txt"
git push