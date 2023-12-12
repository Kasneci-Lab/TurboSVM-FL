# bash code for reproducing our experiments
for project in femnist celeba shakespeare # covid19
do
    for seed in 0 1 2 3 4
    do
        e=1
        for c in 8 16 32
        do
            python main_FL.py -p $project -seed $seed -fl FedAvg -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedAdam -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedAMS -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedProx -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl MOON -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedAwS -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl Ours -C $c -E $e
        done

        c=8
        for e in 2 4
        do
            python main_FL.py -p $project -seed $seed -fl FedAvg -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedAdam -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedAMS -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedProx -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl MOON -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl FedAwS -C $c -E $e
            python main_FL.py -p $project -seed $seed -fl Ours -C $c -E $e
        done
    done
done