# Conditional-seq2seq-VAE
Lab homework of 2021 spring Deep Learning and Practice class in NYCU


Build a conditional seq2seq VAE model in this lab homework to achieve two goals: 

1. Given a verb and its tense and predict its tenses transformations with another given tense.
2. Generate verbs with 4 tenses(simple tense, third-person singular, present progressive tense, and past tense) by Gaussian noise.


Notice that 
1. Use LSTM as the Recurrent Neural Network layer in Encoder and Decoder.
2. Adopt teacher forcing to help the model to learn.
3. Monotonic KL annealing is used to alleviate the KL-vanishing . The KL-weights of each epoch are shown in the following figure.


The parts of results are shown below. BLEU score and gaussian score is used to evaluate the result of prediction (result_prediction.txt) and the result of generation (result_generation.txt), respectively. See details in utils.py.

![image](https://user-images.githubusercontent.com/69135204/124858643-0afe3680-dfe1-11eb-9359-e9626fac3474.png)

