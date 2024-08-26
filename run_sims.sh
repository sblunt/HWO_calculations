sigma_true=2 # run sims for 2, 0.5, 0.1
n_stars=(15 25 50 60)
age_unc=(0.2 1 5)
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