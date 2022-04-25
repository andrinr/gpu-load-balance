rm out/measurements.csv
touch out/measurements.csv


for j in $(seq 1 $1); do 
    for c in 500, 1000, 1500, 2000; do 
        echo $j
        echo $c 
        mpirun -np $j build/final_program $c 1024
    done
done