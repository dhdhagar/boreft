#!/bin/bash -e
# Job wrapper used by the experiment launchers. Runs python from the
# environment that submitted the job. Activate that environment first.
printf "Starting run on $(hostname)\n\n"
printf "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}\n\n"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
printf "Running command:\npython"
printf " %q" "$@"
printf "\n\n"
python "$@"
