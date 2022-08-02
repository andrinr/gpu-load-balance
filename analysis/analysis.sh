
rm *.out
rm results*

# No optimizatoin
for j in 30; do
    sed -i  "s/job-name=.*$/job-name=\"orbit-0-$j\" /g" analysis.job.sh;
    sed -i "s/pParticles=.*$/pParticles=$j/g" analysis.job.sh;
    sed -i "s/pDomains=.*$/pDomains=10/g" analysis.job.sh;
    sed -i "s/opt=.*$/opt=0/g" analysis.job.sh;
    sed -i "s/outFile=.*$/outFile=\"results0\"/g" analysis.job.sh;
    # run script
    sbatch --wait analysis.job.sh
done


# No optimizatoin
for j in 30; do
    sed -i  "s/job-name=.*$/job-name=\"orbit-1-$j\" /g" analysis.job.sh;
    sed -i "s/pParticles=.*$/pParticles=$j/g" analysis.job.sh;
    sed -i "s/pDomains=.*$/pDomains=10/g" analysis.job.sh;
    sed -i "s/opt=.*$/opt=1/g" analysis.job.sh;
    sed -i "s/outFile=.*$/outFile=\"results1\"/g" analysis.job.sh;
    # run script
    sbatch --wait analysis.job.sh
done
