# DeepLearning RNN
This Repo representing another popular neural network,Recurrent Neural Networks (RNNs), which are the natural extention of feedforward networks to understanding the input / output relationship between ordered sequences.

For example, here is the time sequence data, which represents 140 days Apple stock price. Using RNNs to regress the train data, learning the relationship between ordered sequences: time and stock price. And the RNNs model can be used to make prediction on the future sequences.

<img width="961" alt="apple_rnn_prediction" src="https://user-images.githubusercontent.com/36088488/39732428-e464f98a-5232-11e8-9a12-d1fe10a8b206.png">


There are two folders, 

- RNNs_keras folder represents a project using tensorflow,keras to implement RNNs. 
- RNN_fromscratch folder contains building RNNs from scratch based on matrix operation(using numpy), and using the autograd API.

There is a simply implement of [simple RNN](https://github.com/BrownTian/DeepLearning_RNN/blob/master/RNN_fromscratch/RNN.ipynb) from scratch to deal with time sequence data, and this architecture can perform very well.

Apple Stock Example:

<img width="1425" alt="2018-05-29 12 36 21" src="https://user-images.githubusercontent.com/36088488/40639791-705706b2-62d8-11e8-872c-3b98db3000b2.png">

<img width="1410" alt="2018-05-29 12 36 34" src="https://user-images.githubusercontent.com/36088488/40639792-70782d74-62d8-11e8-9c0b-2771ecfcce6b.png">


If you have any question or comments, please feel free to contact me: bowentian2017@u.northwestern.edu 
The material in this repository is not to be distributed, copied, or reused without written permission from the author.
