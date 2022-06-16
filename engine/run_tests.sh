#!/bin/sh

echo Enter the number of folds and the model that you would like to use:

read folds
read model

echo ------- RESULTS --------

for (( fold=0; fold<$folds; fold++ ))
do 
    python -W ignore train.py --fold $fold --model $model
done

