rm out/measurements.csv
touch out/measurements.csv

echo "time, N, np" >> out/measurements.csv


for j in $(seq 1 $1); do 
    for c in 2000, 4000, 8000, 16000; do 
        mpirun -np $j build/final_program $c 1024
    done
done