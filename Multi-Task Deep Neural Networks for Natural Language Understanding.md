# Multi-Task Deep Neural Networks for Natural Language Understanding

**Thesis**: Using multitask learning in combination with pre-trained models is a potent combination.

**Abstract:** We obtain a state-of-the-art on 8 out of 9 of the GLUE tasks by learning representations of language across multiple tasks. This approach beats the performance of a pretrained language model alone. We demonstrate that this approach also permits substantial domain adaptation, which permits learning with substantially smaller datasets. 



## Introduction

Learning vector-space representation of text is a fundamental need for many NLU tasks. Two ways that people commonly achieve new results on NLU tasks is via *multi-task training* or *pretrained language models*. They want to combine these to see if they mesh well together.

They do this by keeping the lower layers to be general (ie. the text encoding layers), while training the downstream layers to be task-specific. This combines several NLU tasks such as single-sentence classification, pairwise text classification, text similarity, and relevance ranking. The key here isn't several tasks of the same type, but rather **a diversity of tasks**.

This has two distinct advantages:

- It results in a 1.8% absolute GLUE benchmark, meaning increased performance over BERT.
- It results in needing substantially less data to perform better, since adversarial structure may be encoded in the other tasks.

## Tasks

**Single Sentence Classification**: You get one sentence, and you match it to a categorical label. For example, sentiment analysis.

**Text Similarity:** It's a regression task, you get two sentences, and you want to output a real-valued number indicating the semantic similarity.

**Pairwise Text Classification**: Given a pair of sentences, understand the relationship between them. For example, *RTE* and *MNLI* are two tasks where you try and understand the label. The task is to understand whether they *contradict*, *entailment*, or *neutral* with respect to each other.

**Relevance Ranking**: Given a query and a list of candidate answers, rank the candidate answers in order of relevance with respect to the query. Think of Google Search or *QNLI*.

## The Proposed DNN Model

I think the [DNN model figure](https://i0.wp.com/syncedreview.com/wp-content/uploads/2019/02/image-3-1.png?w=1018&ssl=1) summarizes this the best in the fewest words.	

### Training Procedure

