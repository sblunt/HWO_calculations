mu_true=13.0
sigma_true=0.001
n_stars=(1 3 5 10 20 30 60)
age_unc=(0.2 0.5 0.7 1)
for i in ${n_stars[@]};
do
    for j in ${age_unc[@]}; 
    do
        python likelihood.py $mu_true $sigma_true $i $j & 
    done
done

echo "Running all scripts"
wait # wait until all scripts finish
echo "All scripts finished"