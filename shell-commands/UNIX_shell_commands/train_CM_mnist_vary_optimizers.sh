optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=5e-1
    elif [ "$optim" = "ngd" ]; then
        lr=2
    fi

    python train.py --model CM --optimiser $optim --lr $lr --num_workers 64 --batch_size 4096 --num_epochs 30 --CMKS 4 --CMHL 4
done

