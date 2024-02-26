hidden_nodes=(512 1024 2048)
hidden_layers=2

for HN in "${hidden_nodes[@]}"
    do 
        python train.py --model LM --LMHN $HN --LMHL $hidden_layers
        #echo $HL
    done