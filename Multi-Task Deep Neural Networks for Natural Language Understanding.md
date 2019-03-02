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
<img src="https://i0.wp.com/syncedreview.com/wp-content/uploads/2019/02/image-3-1.png"
     alt="Results"
     style="float: left; margin-right: 10px;" />
     

### Training Procedure

MT-DNN was trained by two stages: first, a general language model was pretrained, followed by multi-task fine tuning. The LM pretraining was done by Devlin et al in BERT. 

In order to fine tune ot the tasks, we select a mini-batch by merging a fraction of each task, and then shuffle the dataset. Finally, we train the model according to each dataset's loss function (read: multiple, differing loss functions) using SGD.

Loss functions:

- For classification, they used cross-entropy.
- For textual similarity, they use MSE since it's a regression task.
- For learning-to-rank, they minimized negative log-likelihood.

## Experiments
*Datasets*: they provide a short summary of each dataset in this section. These are summarized below.

- **GLUE**: collection of nine NLU tasks, which is commonly used to evaluate generalization and robustness of NLU models.
- **CoLA**: "Is this English sentence linguistically acceptable?", Label: real-valued.
- **SST-2**: What is the sentiment?, Label: real-valued
- **STS-B**: "How similar are two sentences?", label: one-to-5 by a human annotator.
- **QNLI**: "Is this a correct answer to my query", binary labels
- **QQP**: "Are two quetions semantically equivalent?", label: binary classification
- **MRPC**: semantic equivalence, binary labels.
- **MNLI**: textual entailment, multiclass classification task.
- **RTE**: binary entailment task
- **WNLI**: reading comprehension task, "who/what does the pronoun refer to?"
- **SNLI**: entailment task with multiclass labels.
- **SciTail**: entailment tas with MultiClass labels. 

### Implementation Details
They used HuggingFace's implementation of BERT with Adamax optimizer. Wordpieces were used to provide subword tokenization, with a token length of 512. Gradients were clipped to 1.

### Results
See Table 3 for GLUE results.

The tl;dr is that MT-DNN consistently beat every other model, where the comparison models are BERT-base, MT-DNN, and ST-DNN, which only uses a single task. The difference between ST-DNN is the design of a task-specific output model, for example, SAN.

### Domain Adaptation
<img src="https://www.groundai.com/media/arxiv_projects/506191/fig/da.png"
     alt="Results"
     style="float: left; margin-right: 10px;" />
     
[This graph](https://www.groundai.com/media/arxiv_projects/506191/fig/da.png) describes the improvement domain adaptation does to learning new tasks. We see that this significantly helps provide better results given some pretraining.

They split the training data of SNLI and SciTail and trained on 0.1%, 1%, 10% and 100% of the training data. We can see that MT-DNN always requires less training data than BERT does, since it benefits from multi-task learning.


# Conclusion
This work combined BERT with multi-task learning and showed significant results. I fully expect this to be duplicated in other NLP subfields.