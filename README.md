# Conditional-seq2seq-VAE
Lab homework of 2021 spring Deep Learning and Practice class in NYCU


Build a conditional seq2seq VAE model in this lab homework to achieve two goals: 

1. Given a verb and its tense and predict its tenses transformations with another given tense.
2. Generate verbs with 4 tenses(simple tense, third-person singular, present progressive tense, and past tense) by Gaussian noise.


The objective function is $$E_{z \sim q(z|x;\theta^{'})} \log\mathrm{P}(x|z;\theta) -KL(q(z|x;\theta^{'}) || \mathrm{P}(z;\theta))$$
