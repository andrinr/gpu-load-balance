
for j in $(seq 0 $1); do 
    rm measurements$((2**$j)).csv
    touch out/measurements$((2**$j)).csv

    echo "time, N, np" >> out/measurements$((2**$j)).csv

    for c in 4000, 8000, 16000, 32000; do 
        mpirun -np $((2**$j)) build/final_program $c 1024 out/measurements$((2**$j)).csv
    done
done
