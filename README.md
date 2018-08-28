# Tweet-Classification-Deep-Learning-Traffic
## Project Summary
In this project, I used deep learning architectures including Convolutional and Recurrent Neural Network to classify tweets into three
labels: 1) Non-Traffic (NT), 2) Traffic Indcident (TI) ==> Examples: traffic crashes, disabled vehicles, highway maintenance, work zones,
road closure, and vehicle fire, 3) Traffic Conditions and Information (TCI). Examples: daily rush hours, traffic congestion, 
traffic delays due to high traffic volume, and jammed traffic.

Using Twitter Search API tweets, I collected 51,100 tweets and then manually labeled them. In traffic domain, this is by far the largest 
traffic-based tweet labeled. All labeled tweets and their association codes are provided in the Dataset folder. 

First, tweets are modeled through word embedding tools using two models: (1) Word2vec, (2) FastText.
Afterward, the created word embedding matrix is fed into deep learning architectures to classify tweets into three labels. 

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

## Paper Abstract
"In recent years, several studies have harnessed Twitter data for detecting traffic incidents and monitoring traffic conditions. Researchers have utilized the bag-of-words representation for converting tweets into numerical feature vectors. However, the bag-of-words not only ignores the order of tweet’s words but suffers from the curse of dimensionality and sparsity. A common approach in literature for dimensionality reduction is to build the bag-of-words on the top of pre-defined traffic keywords. The immediate criticisms to such a strategy are that the pre-defined set of keywords may not include all traffic keywords and the tweet language is subjected to change over time. To address these shortcomings, we utilize the power of deep-learning architectures for both representing tweets in numerical vectors and classifying them into three categories: 1) non-traffic, 2) traffic incident, and 3) traffic information and condition. First, we map tweets into low-dimensional vector space through word-embedding tools, which are also capable of measuring the semantic relationship between words. Supervised deep-learning algorithms including convolutional neural network (CNN) and recurrent neural network (RNN) are then deployed on the top of word-embedding models for detecting traffic events. For training and testing our proposed model, a large volume of traffic tweets is collected through Twitter API endpoints and labeled through an efficient strategy. Experimental results on our labeled dataset show that the proposed approach achieves clear improvements over state-of-the-art methods."




