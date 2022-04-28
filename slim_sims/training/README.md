## Slim Model Training

The first major batch of simulations was run with this:

    python ../../tools/slurm_slim_runner.py generate  --slim '~/src/SLiM_build/slim' \
       --max-array 5000 --num-files 10 \
       --secs-per-job 10 --batch-size 30 --dir '../../data/slim_sims/' --seed 12 \
       --script slurm.sh ./segment_logL_logrbp_logrf_wide.json

which generates numerous `slurm_0.sh`, `slurm_1.sh`, etc. scripts given the limited 
`--max-array` size of Slurm array jobs that can be submitted. These could be launched
by hand, but the script also writes a `runner_script.sh` which submits a job, 
monitors the cluster's status, and submits another if there's nothing coming out of
`squeue`. It took a few runs to get everything done; importantly, each time one must 
remove all the `script_*.sh` file beforehand, as there will be fewer since existing 
simulation results won't be re-run and clobbered.
