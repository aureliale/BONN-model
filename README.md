# Options discovery with Budgeted Reinforcement Learning
(under review as a conference paper at ICLR 2017)

Aurélia Léon, Ludovic Denoyer

We consider the problem of learning hierarchical policies for Reinforcement Learning able to discover options, an option corresponding to a sub-policy over a set of primitive actions. Different models have been proposed during the last decade that usually rely on a predefined set of options. We specifically address the problem of automatically discovering options in decision processes. We describe a new RL learning framework called Bi-POMDP, and a new learning model called Budgeted Option Neural Network (BONN) able to discover options based on a budgeted learning objective. Since Bi-POMDP are more general than POMDP, our model can also be used to discover options for classical RL tasks. The BONN model is evaluated on different classical RL problems, demonstrating both quantitative and qualitative interesting results.

## Links

[OpenReview](http://openreview.net/forum?id=H1eLE8qlx)

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

## Bibex