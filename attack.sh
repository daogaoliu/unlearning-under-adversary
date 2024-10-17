# Random noise + optimzied
for T in 1 #2 
    do
    for adv_lr in 0.2 # 0.5 0.01 0.02 0.05 
        do
        for n in 101  #num of advX
        do
            for t in 11 # num of unlearning step
            do
                python unlearning-cifar10.py --optimize_images --unlearn_n $n --unlearn_epochs $T --step_optimized $t --adv_lr $adv_lr --unlearn_lr 0.05 --unlearn_method 'ga' --attack_method 'black_box'
            done 
        done
    done
done