optimizers=(sgd ngd)

for optim in "${optimizers[@]}"
do 
    #specify different lr for sgd or ngd
    if [ "$optim" = "sgd" ]; then
        lr=8e-2
    elif [ "$optim" = "ngd" ]; then
        lr=5e-2
    fi

    python train.py --model CM --optimiser $optim --lr $lr --num_workers 8 --batch_size 4096 --num_epochs 20 --CMHL 0
done

