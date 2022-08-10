#!/bin/bash

if [[ $# > 1 ]]; then
    echo "Reseting $1"
    mkdir logs/"$1-RESET"
    mkdir models/"$1-RESET"
    cp models/$1/config.yml models/"$1-RESET"

    nohup python3 main.py $1 "reset" > logs/"$1-RESET"/logs.txt &
else
    echo "Starting fresh trainig $1"
    mkdir logs/$1
    mkdir models/$1
    cp config.yml models/$1

    nohup python3 main.py $1 > logs/$1/logs.txt &
fi