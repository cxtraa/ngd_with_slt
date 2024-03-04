# Define new selection criteria
model="LM"
optimisers=("sgd" "ngd")  # Corrected variable name
lmhn=8
lmhl=2
batch_size=512

# Convert Bash array to a JSON-like string for optimisers
optimisers_str=$(printf '"%s",' "${optimisers[@]}")
optimisers_str="[${optimisers_str%,}]"  # Remove trailing comma and wrap in brackets

# Define RLCT hyperparameters
num_draws=1000
num_chains=2
epsilon=1e-5
gamma=1

# Call the Python script with the new parameters
python experiments/mnist/eval_mnist_optimisers.py \
    --criteria "{\"model\":\"$model\", \"optimiser\":$optimisers_str, \"LMHN\":$lmhn, \"LMHL\":$lmhl, \"batch_size\":$batch_size}" \
    --num_draws $num_draws \
    --num_chains $num_chains \
    --epsilon $epsilon \
    --gamma $gamma


