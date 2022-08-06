#!/bin/bash
mkdir logs/$1

mkdir models/$1
cp config.yml models/$1

nohup python3 main.py $1 > logs/$1/logs.txt &