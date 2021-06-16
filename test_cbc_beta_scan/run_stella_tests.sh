#!/bin/bash

for subfolder in `ls stella* -d`
do
  echo $subfolder
  #script=$subfolder + "/run_gs2.sh"
  #echo $script
  cd $subfolder 
  sbatch "run_stella_marconi.sh"
  cd ..
  sleep 0.1
done
