
for j in $(seq 1 $1); do 
    rm out/measurements$j.csv
    touch out/measurements$j.csv

    echo "time, N, np" >> out/measurements$j.csv

    for c in 4000, 8000, 16000, 32000; do 
        mpirun -np $j build/final_program $c 1024
    done
done