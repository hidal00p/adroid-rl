#!/bin/bash
FILE="log$1.txt"
nohup python3 main.py > logs/$FILE &