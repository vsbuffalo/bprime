for i in {1..50}
do
snakemake -j 999 --rerun-incomplete --use-conda --config config="./cluster_talapas.json" --cluster-config cluster_talapas.json \
   --jobname='bgs_segment.{rulename}.{jobid}.sh' --scheduler greedy --rerun-incomplete \
   --cluster 'sbatch --error {cluster.error} --output {cluster.output} -A {cluster.account} -p {cluster.partition} -n {cluster.n} -t {cluster.time} --mem-per-cpu {cluster.mem} -c {cluster.cores}' \
   --batch all=$i/50 all
done
