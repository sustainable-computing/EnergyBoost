#!/bin/bash
for f in process_*; do 
	sbatch "$f"
done
