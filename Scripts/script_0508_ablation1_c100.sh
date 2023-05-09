for seed in 1 2 3 4; do
    for act_prob in 0.01; do
        for dist in 0.0 0.3; do 
            for task in "CIFAR100"; do
                for mode in "fedavg"; do
                    python BL_0407_GU.py --mode $mode --task $task --distribution $dist --act_prob $act_prob --seed $seed --GU 111.06
                    python BL_0407_GU.py --mode $mode --task $task --distribution $dist --act_prob $act_prob --seed $seed --GU 111.10
                done
            done
        done
    done
done