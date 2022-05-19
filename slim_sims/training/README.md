## Slim Model Training

The first major batch of simulations was run with this:

    python ../../tools/slurm_slim_runner.py generate  --slim '~/src/SLiM_build/slim' \
       --max-array 1000 --num-files 10 \
       --secs-per-job 10 --batch-size 30 --dir '../../data/slim_sims/' --seed 12 \
       --script slurm.sh ./segment_logL_logrbp_logrf_wide.json

which generates numerous `slurm_0.sh`, `slurm_1.sh`, etc. scripts given the limited 
`--max-array` size of Slurm array jobs that can be submitted. These could be launched
by hand, but the script also writes a `runner_script.sh` which submits a job, 
monitors the cluster's status, and submits another if there's nothing coming out of
`squeue`. It took a few runs to get everything done; importantly, each time one must 
remove all the `script_*.sh` file beforehand, as there will be fewer since existing 
simulation results won't be re-run and clobbered.


Next we run the `trees2data.py` tool to take the treeseqs, recapitate them,
estimate diversity, extract features from the metadata, and then write the results.

     python ../../tools/trees2data.py --outfile segment_logL_logrbp_logrf_wide_data.npz \
       --ncores 50 --features 'mu,s,L,rbp,rf,h,rep'  ../../data/slim_sims/segment_logL_logrbp_logrf_wide/

## Fitting the DNNs

Uses the `Snakemake` file in this directory, with the appropriate JSON config file:

    bash snakemake_runner.sh -c ./segment_logL_logrbp_logrf_wide.json

This takes the `.npz` raw simulation data, converts it to a `LearnedFunc` 
object, which manages all the details needed to do test/train splits, 
check the input feature matrix. This data manipulation and the actual
DNN fitting with tensorflow/keras is done with `tools/fit_sims.py [data | fit]`.

## DNN B' Maps

The `data/dnnb` directory was created with:

    python ../../bgspy/command_line.py dnnb-write --dir ../../data/dnnb \
      --recmap ../../data/annotation/hapmap_genetic_map.txt --annot ../../data/annotation/conserved_slop.bed.gz \
     --seqlens ../../data/annotation/hg38_seqlens.tsv --max-map-dist 0.1 \
     ../../data/slim_sims/segment_logL_logrbp_logrf_wide/fits/segment_logL_logrbp_logrf_wide/segment_logL_logrbp_logrf_wide_0n128_0n64_0n32_0n8_2nx_eluactiv_fit_0rep


 python -mpdb ../../bgspy/command_line.py dnnb-write --w '1e-7,1e-8,1e-9' --t '0.1,0.01,1e-3,1e-4,1e-5,1e-6' --dir ../../data/dnnb --recmap ../../data/annotation/hapmap_genetic_map.txt --annot ../../data/annotation/conserved_slop.bed.gz --seql
ens ../../data/annotation/hg38_seqlens.tsv --max-map-dist 0.1 ../../data/slim_sims/segment_logL_logrbp_logrf_wide/fits/segment_logL_logrbp_logrf_wide/segment_logL_logrbp_logrf_wide_0n128_0n64_0n32_0n8_2nx_eluactiv_fit_0rep

Then we run prediction across each of the chunks, across our SLURM cluster.
I do this with:

    ./snakemake_runner.sh -c segment_predict.json -l cluster_talapas_predict.json -s tools/predict_snakefile


## new stuff

python ../../tools/slurm_slim_runner.py generate  --slim '~/src/SLiM_build/slim' --max-array 1000 --num-files 10  --secs-per-job 10 --batch-size 30 --dir '../../data/slim_sims/' --seed 12  --script slurm.sh ./bmap_hg38_reps.json 

bash snakemake_runner.sh -c ./bmap_hg38.json  -r data

python  ../../bgspy/command_line.py dnnb-write --w '1e-7,1e-8,1e-9' --t '0.1,0.01,1e-3,1e-4,1e-5' --dir ../../data/dnnb --recmap ../../data/annotation/hapmap_genetic_map.txt --annot ../../data/annotation/conserved_slop.bed.gz --seqlens ../../data/annotation/hg38_seqlens.tsv --max-map-dist 0.1 ../../data/slim_sims/bmap_hg38_reps/fits/bmap_hg38_reps_0n128_0n64_0n32_0n8_2nx_eluactiv_fit_{0,1,2,3}rep

./snakemake_runner.sh -c bmap_hg38_reps_predict.json -l cluster_talapas_predict.json -s predict_snakefile 
