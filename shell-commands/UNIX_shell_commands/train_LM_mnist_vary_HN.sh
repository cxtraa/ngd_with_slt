hidden_nodes=(2 4 8)
hidden_layers=2

for HN in "${hidden_nodes[@]}"
    do 
        python train.py --model LM --LMHN $HN --LMHL $hidden_layers --num_workers 12 --batch_size 128 --num_epochs 5
        #echo $HL
    done