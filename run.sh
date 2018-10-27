#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'c' ]; then
  $PY cap_finder.py
elif [ $1 = 's' ]; then
  # $PY sim.py
  $PY sim_wtrace_exp.py
else
  echo "Arg did not match!"
fi