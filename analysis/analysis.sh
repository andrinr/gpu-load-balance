
rm *.out
rm results

# No optimizatoin
for j in 25; do
    sed -i  "s/job-name=.*$/job-name=\"orbit-0-$j\" /g" analysis.job.sh;
    sed -i "s/nParticles=.*$/nParticles=j/g" analysis.job.sh;
    sed -i "s/pDomains=.*$/nParticles=10/g" analysis.job.sh;
    sed -i "s/opt=.*$/nParticles=0/g" analysis.job.sh;
    ed -i "s/outFile=.*$/outFile=\"results0\"/g" analysis.job.sh;
    # run script
    sbatch --wait analysis.job.sh
done


# No optimizatoin
for j in 25; do
    sed -i  "s/job-name=.*$/job-name=\"orbit-1-$j\" /g" analysis.job.sh;
    sed -i "s/nParticles=.*$/nParticles=j/g" analysis.job.sh;
    sed -i "s/pDomains=.*$/nParticles=10/g" analysis.job.sh;
    sed -i "s/opt=.*$/nParticles=1/g" analysis.job.sh;
    sed -i "s/outFile=.*$/outFile=\"results1\"/g" analysis.job.sh;
    # run script
    sbatch --wait analysis.job.sh
done
