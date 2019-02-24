# Stochastic Answer Networks for Natural Language Inference

**Hypothesis:** it's hard to model the textual entailment between two sentences in one pass, so we examine two sentences iteratively (via a *recurrent state*), in order to model a better relation.

## Motivation
- You have some **premise** (*Obama won the 2016 election*), and some **hypothesis** (*The Trump presidency from 2016-2020 was fantastic*), and you want to understand how these two sentences are related.
- Most of NLI currently uses a single step to compute this process, and instead, this paper proposes to maintain a state and iteratively refine the predictions for that state.

## Enter: SAN
- Let's say you have some premise P, which consists of *m* words, and some hypothesis H, which consists of *n* words. You want to find some logical relationship between *P* and *H*, modeled as *R*.
- We can develop some recurrent state s_t, which processes multiple passes through P and H until some predetermined t=T steps.
- The overall architecture is Siamese.
- We do this via four layers:
	1. **Lexicon Encoding Layer:** this is a concatenation of GloVe embeddings and character embeddings (for OOV words). FFN(x) = W_2*ReLU(W_1x + b_1) + b_2 is used as a transformation of these input embeddings to develop a lexical analysis. 
		- Imagine the FFN as two layers of single-point convolutions; it helps learn a better representation of the text.
		- Output Shape: *d* x *m*, *d* x *n*.
	2. **Contextual Encoding Layer:** this layer stacks two BiLSTMs to encode the context for each word in P and H. The output of each LSTM has a maxout activation, followed by a concatenation together. 
		-  Output Shape: C^P: *2d* x *m*, C^H: *2d* x *n*.
	3. **Memory Layer:** Start with a linear transformation of C^P and C^H, and then do A = attention(C^P, C^H). Finally, apply the attention to each premise and hypothesis on each tower: U^P = concat(C^P, C^H*A), and the converse for hypothesis. Lastly, generate the memory by using a BiLSTM on everything you know: M^P = BiLSTM([U^P, C^P]) = BiLSTM([C^P, C^H * A, C^P]). It helps to sandwich the hypothesis since you're using a BiLSTM.
	4. **Answer Layer:** This is the stochastic part of the paper, and computes a relation over *T* timesteps. They apply layer-level dropout and average the T outputs from each timestep, which ensures the model doesn't rely too much on any particular layer. Here's a couple of important notes:
		- M^P = softmax(s_{t-1} * W_5 * M_p) * M_j^P
		
## Experiments

**Dataset**: They evaluate on [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/), [SNLI](https://nlp.stanford.edu/projects/snli/), and [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) dataset.

**Implementation Details:** used spaCy and PyTorch. This is a lot of hyperparameter details, so not worth repeating.

### Results:
Table 2 in the paper demonstrates the results.
