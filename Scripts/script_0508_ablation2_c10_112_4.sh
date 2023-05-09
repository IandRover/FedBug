for seed in 1 2 3; do
    for act_prob in 0.01; do
        for dist in 0.3; do 
            for task in "CIFAR10"; do
                for mode in "fedavg"; do
                    python BL_0407_GU.py --mode $mode --task $task --distribution $dist --act_prob $act_prob --seed $seed --GU 112.04 --epoch 10
                done
            done
        done
    done
done
