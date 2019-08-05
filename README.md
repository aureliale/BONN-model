# Budgeted Hierarchical Reinforcement Learning
(IJCNN 2018)

Aurélia Léon, Ludovic Denoyer

In hierarchical reinforcement learning, the framework of options models sub-policies over a set of primitive actions. In this paper, we address the problem of discovering and learning options from scratch. Inspired by recent works in cognitive science, our approach is based on a new budgeted learning approach in which options naturally arise as a way to minimize the cognitive effort of the agent. In our case, this effort corresponds to the amount of information acquired by the agent at each time step. We propose the Budgeted Hierarchical Neural Network model (BHNN), a hierarchical recurrent neural network architecture that learns latent options as continuous vectors. With respect to existing approaches, BHNN does not need to explicitly predefine sub-goals nor to a priori define the number of possible options. We evaluate this model on different classical RL problems showing the quality of the resulting learned policy.

## Links

(http://www.smart-labex.fr/publications/pdf/5c1bde322f26c.pdf)

## Requirements

* python
* numpy
* scipy
* tensorflow
* gym (environments are coded as in gym to be compatible)

## Execution

To run the BONN model with defaults parameters:
```
python  run.py --env rooms2 --model BONN --use_x True --num_hidden 5 --representation_x linear:5:relu --representation_y linear:5:relu --lr 0.01 --lambd 0.5
```

To run the Discrete-BONN model with defaults parameters:
```
python  run.py --env rooms2 --model D-BONN --nH 9 --use_x True --num_hidden 5 --representation_x linear:5:relu --representation_y linear:5:relu --lr 0.01 --lambd 0.5
```

To run the recurrent policy gradient (with one GRU cell):
```
python  run.py --env rooms2 --model GRUReinforce --num_hidden 5 --representation_x linear:5:relu  --lr 0.01
```


## Others parameters
Here are all the parameters:
```
--log {csv or console} [to save results]
--directory [if log:csv, directory where to save results]
--filename [if log:csv, name to save files of experiment]

--env {rooms2, rooms4, rooms9, rooms2stocha, rooms4stocha, rooms9stocha, cartpolestocha, or gym environment name} [name of environment to use; can be a gym one]
--stocha [stochasticity level if stochastic environment]

--model {GRUReinforce,BONN,D-BONN} [model to use]
--lambd [cost for BONN and D-BONN]
--use_x [if BONN or D-BONN : if use_x=True, for environments "rooms", the position of the agent will be observation x; else all observations will be observations y]
--num_hidden [for hidden cell in recurrent network]
--nH [for D-BONN, number of discrete policies]
--representation_x 
--representation_y 
--exploration_epoch_min [number of epoch when we use a cost=0 for BONN and DBONN]
--exploration_epoch_nb [number of epoch since exploration_epoch_min to increase the cost]

--gamma [discount]
--lr [learning rate]
--n_batch [number of trajectories used to learn]
--iter_max 
--nb_iter_for_obs 
--obs_each_n_trajs
```
