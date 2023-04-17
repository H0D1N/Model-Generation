# Model-Generation
The code for flexible model generation

Use the Colossal-AI for training

## start training with
~~~
colossalai run --nproc_per_node 4 train.py
~~~
### Using EMA for trainning
~~~
colossalai run --nproc_per_node 4 train.py --model_ema --world_size 4
~~~

## Problem
In config.py, Gradient Clipping is not working，so I comment out 
~~~
#clip_grad_norm = 1.0
~~~

## Gate Selection Network
![6211679574347_.pic.jpg](..%2F..%2FLibrary%2FContainers%2Fcom.tencent.xinWeChat%2FData%2FLibrary%2FApplication%20Support%2Fcom.tencent.xinWeChat%2F2.0b4.0.9%2F830026af28dd45ab7f702f555756a7f3%2FMessage%2FMessageTemp%2Fd2f1a1b506b075144a33938b8cc9f13a%2FImage%2F6211679574347_.pic.jpg)