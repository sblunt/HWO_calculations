sigma_true=1
n_stars=(10 25 50 100)
age_unc=(2 1 0.5 0.1)
for i in ${n_stars[@]};
do
    for j in ${age_unc[@]}; 
    do
        python likelihood.py $sigma_true $i $j & 
    done
done

echo "Running all scripts"
wait # wait until all scripts finish
echo "All scripts finished"