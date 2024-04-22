# CS599 Final Project 

## Setup

The code requires [Anaconda](https://www.anaconda.com/download).

Please create a virtual environment before running the code (see documentation for [Visual Code](https://code.visualstudio.com/docs/python/environments))

To install all dependencies run the following commands in a terminal:
```
cd code
pip install -r requirements.txt
```

### Usage
The file `main.py` contains 3 methods: `plot_all_results()`, 
`main()`, and `ablation()`. Please uncomment the method you wish to run.

### Experiments
Our agent is defined as the class `AdvancedQLearner`. This agent is capable of different combinations of using UCB1
or epsilon-greedy exploration, and TD(Lambda) or TD(1) for return targets. For the results presented in the fianl report,
we use TD(1) targets and UCB1.

The `main()` method is used to run experiments that train the agents. This method
uses the command line arguments defined at the top of the file to specify hyperparameters. 
For example, to train an Q-Learning agent with UCB1 for exploration and single-step returns on hard_0, 
run the following command from the `code/` directory: 

```python
python -m main --rooms_instance=hard_0 --exploration_strategy='ucb' --lr=0.1 --train_steps=500
```

Please see the `parse_args()` function in `main.py` for a full list of command line arguments. Other hyperparameters such 
as the exploration constant and discount factor are set directly in the `main()` method.

### Ablation
To run the ablation presented in the 2nd intermediate project report, please uncomment the `ablation()` method call and 
comment out the other two methods. Ignores all cmd-line args.

### Plotting Results
To visualize the results presented in the final project report, please uncomment `plot_all_results()` and comment out 
the other two methods. Ignores all cmd-line args.


### Hardware
Code was tested on a desktop running Ubuntu 20.04 LTS with a Ryzen 7900X, RTX 3090 GPU, and 64GB of memory. The minimum hardware
requirements are any machine with a CPU and sufficient memory. Dedicated NVIDIA GPU is not required. The seed is set randomly
upon environment initialization. However, any fixed seed will work as well. 