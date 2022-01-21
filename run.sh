#!/bin/bash

input_path="data/"
output_path="results/"
loss=$1

if [[ -z "$loss" ]]
then
  echo "Please specify loss option: (1). bce; (2). dmt"
else
  ./train.py --option binary -i $input_path -o $output_path -l $loss -r 0.001 -d saw -n 50 -p 10 --thld 30 --early-stop
fi

