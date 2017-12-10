#! /bin/bash
#

for i in $(seq 1 20); do
     qsub -v LEARNING_RATE=$1 -v RUN_DIRECTORY="different_layers_$i" run-different_layers.sub
done
