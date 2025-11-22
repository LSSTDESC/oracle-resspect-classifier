#!/usr/bin/env bash

pip freeze > requirements.txt
git add requirements.txt
git commit -m "updated requirements.txt"
git push