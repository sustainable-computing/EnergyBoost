#!/bin/bash
for i in {1..70}
do
   echo "Submitting job: $i"
   sbatch process"$i".sh
done
