# Model-Generation
The code for flexible model generation

Use the Colossal-AI for training

## start training with
~~~
colossalai run --nproc_per_node 4 train.py
~~~
## Problem
In config.py, Gradient Clipping is not working，so I comment out 
~~~
#clip_grad_norm = 1.0
~~~