# Tweet-Classification-Deep-Learning-Traffic

In this project, I used deep learning architectures including Convolutional and Recurrent Neural Network to classify tweets into three
labels: 1) Non-Traffic (NT), 2) Traffic Indcident (TI) ==> Examples: traffic crashes, disabled vehicles, highway maintenance, work zones,
road closure, and vehicle fire, 3) Traffic Conditions and Information (TCI). Examples: daily rush hours, traffic congestion, 
traffic delays due to high traffic volume, and jammed traffic.

Using Twitter Search API tweets, I collected 51,100 tweets and then manually labeled them. In traffic domain, this is by far the largest 
traffic-based tweet labeled. All labeled tweets and their association codes are provided in the Dataset folder. 

First, tweets are modeled through word embedding tools using two pre-trained word2vec: 1) Twitter word2vec
provide in https://www.fredericgodin.com/software/  2). Google word2vec provided in https://code.google.com/archive/p/word2vec/ 
Afterward, the create word embedding matrix is fed into deep learning architectures to classify tweets into three labels. 

Three supervised deep learning architectures have used for the classification task: 1) only CNN, 2) only LSTM, 3) CNN+LSTM

Why deep learning for traffic domain tweet classification:
-Problems with bag-of words representation: High-dimensional and sparse vector space
-Ways to address bag-of-words representation: Using statistical feature selection models, and Building bag-of-words representation on
the top of a pre-defined set of traffic-related keywords (the common approach)
-Immediate criticism to this solutio: Selected keywords might not contain all words for all dataset and it is subjected to change over time.
Twitter language contains abbreviation and changes over time
-One possible solution
 Using word embedding tools for tweet modeling and supervised deep learning architectures for classification task
-Advantages:Address the curse of dimensionality and remove sparsity, No need for a pre-defined set of traffic keywords



