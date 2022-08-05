
for j in 2; do
    sed -i  "s/job-name=.*$/job-name=\"cpuReduce-$j\" /g" analysis.job.sh;
    declare -i pow
    pow=$((2**$j))
    sed -i "s/nThreads=.*$/nThreads=$pow/g" analysis.job.sh;
    echo $pow
    # run script
    sbatch --wait analysis.job.sh
done
