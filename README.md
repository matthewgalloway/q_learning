**ENVIRONMENT**
This environment is a small sub set of stations in zone 1 designed to to explore Q-learning outside of AI gym.

Environment is represented by discreet states as below:

![image](https://user-images.githubusercontent.com/52289894/85226089-92283e80-b3cd-11ea-8a16-6b620165ad35.png)

The reward table for the environments is:

![image](https://user-images.githubusercontent.com/52289894/85226101-abc98600-b3cd-11ea-813a-a8f619331494.png)

The environment is designed to see if the agent can learn the difference between the shortest route and the cheapest route

**BEST MODEL**

Policy: e-greedy > UCB

e-greedy hyperparameters from the grid search:
epsilon = 1
decay rate = 0.87
alpha = 0.3
gamma= 0.7
