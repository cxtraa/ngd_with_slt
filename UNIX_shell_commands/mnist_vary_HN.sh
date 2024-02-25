hidden_nodes=16
hidden_layers=(1 2)

for HL in "${hidden_layers[@]}"
    do 
        python train.py --model LM --LMHN $hidden_nodes --LMHL $HL
        #echo $HL
    done