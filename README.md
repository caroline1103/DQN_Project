# Deep Reinforcement Learning: Playing Atari games 🎮

![](images/tele.gif) 

## What is Deep Reinforcement Learning?

Deep Reinforcement learning is a combination of reinforcement learning and deep learning. 
In reinforcement learning, the problem is not supervised, meaning we let the computer explore the environment by itself and only give him a reward signal when he takes a good or bad action. We create a map between states and actions to the rewards they lead to


### Details on a Markov Decision Process


### And what about DQN?


![](images/Atari.jpg) 
## The games we played:

I tried to implement a DQN algorithm on CartPole and Breakout from the gym environment. Using this environment allows us to easily observe and take an action in the game. It allows to easily access the rewards and the next state of the environment.

Both games have a discrete action spaces, Cartpole can only move right or left and Breakout has two more actions available: do nothing and fire. 

## Cartpole

Cartpole before training           |  Cartpole after training
:---------------------------------:|:---------------------------------:
![](visual_examples/Cartpole_before_training.gif)  |  ![](visual_examples/Cartpole_after_training.gif)


## Breakout

Breakout before training           |  Breakout after training (Not obtained with my DQN)
:---------------------------------:|:---------------------------------:
![](visual_examples/breakout_before_training.gif) | <img align="center" width="190" height="240" src="images/not_mine.gif">




## The issues..

<img align="left" width="100" height="150" src="images/pacman.png">
As you can see, the agent did not learn well on the breakout game. 
I wanted to keep the algorithm I used for cartpole as it worked very well to train this game. However, major changes had to be implemented for the algorithm to work on breakout.
The main difference has been the observation space which was no longer a simple vector of length 4 but a full RGB image. Furthermore, cartpole was rewarded at almost each step, as it gained reward from staying pright at each step. However, in breakout rewards are delayed, as it can take multiple frames/states until we hit a brick.
Using the RGB as it is has proven to be difficult, as it's not efficient to train on. The papers proposed to turn the image in a grayscale and to crop it in a 84x84 matrix.
This has proven difficult as I was using torch rather than tanserflow and was running int problem with the tensorflow library which I was not able to fix.
Furthermore, as we observe only one image, we don't have a sense of which direction the ball might be moving in. For this reason, we need to consider a state as four stacked frames, which I was not able to correctly implement as I was running in issues with the the different sizes of the elements.
I did include the notebook I was working on but which I did not manage to finish.


### How to reproduce the results
