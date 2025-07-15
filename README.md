# NPC-l2race
implementing a neuromorphic predictive controller

MLP is type of NN architecture that is foundational in DL. It is simple, classic and powerful when used right
-Data flows forward from input to output (Feedforward).
-Network learns by adjusting Weights using BackProp and Gradient descent
-Activation Functions(ReLu, sigmoid or tanh) introduces non-linearity, thus allowing to learn complex patters
Note:
- Not ideal for sequantial/spatial data (images (CNN),time series or language(use RNN))

MLP is serving as a lightweight policy Net or correction module that will support our Pure Pursuit baseline control.

