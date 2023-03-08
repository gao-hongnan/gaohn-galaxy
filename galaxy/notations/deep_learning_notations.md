# Deep Learning Notations

## NLP Notations

Given a dataframe of five rows:

```markdown
1. This is Foo, how can I help you
2. It might rain today
3. I love football
4. Crazy, Stupid & Love
5. I shot the sheriff
```


We will use this table to explain some terminologies below.

[What is the difference between document, corpus, and vocabulary?](https://stackoverflow.com/questions/72550835/corpus-vs-vocabulary-vs-document-in-nlp).

In NLP, document concept can be a bit vague: a document is a unit, so it can correspond to different text objects, such as entire documents, sentences, passages... In your example, "This is Foo, how can I help you" is a document. "It might rain today" is another document...

A corpus is a collection of documents. In your example, the corpus is composed by 5 documents.

The vocabulary is the list of all the words contained in the corpus, therefore all the words contained in all the documents. Your vocabulary is `[&, can, crazy, foo, football, help, how, i, is, it, love, might, rain, sheriff, shot, stupid, the, this, today, you]`.

```{list-table} Basic NLP Terminologies
:header-rows: 1
:name: basic-nlp-terminologies

* - Notation
  - Description
* - $\mathcal{V}$
  - The set of all unique words in the corpus.
* - $\mathcal{S}$
  - The set of all documents in the corpus (subjected to change).
* - $V$
  - The dimension of the number of vocabulary words in the corpus.
* - Corpus/Collection
  - A collection of documents.
```

````{div} full-width
```{list-table} Transformers and Language Models
:header-rows: 1
:widths: 15 25 5 25 15 15
:class: align-left
:name: transformers-and-language-models

* - Notation
  - Description
  - Category
  - Example
  - References
  - Comments
* - Self-Supervised
  - Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data!
  - Machine Learning Paradigm
  -
  -
  -
* - Transfer Learning
  - The main idea behind transfer learning is that features learned by a model on one task can be reused as a starting point or foundation for learning features on a new task. This can lead to faster training times, better generalization to new data, and improved performance on the new task, especially when the amount of labeled data for the new task is limited.
  - Machine Learning Paradigm
  -
  -
  -
* - GPT-like (also called auto-regressive Transformer models)
  - The dimension of the number of vocabulary words in the corpus.
  - General
  -
  -
  -
* - BERT-like (also called auto-encoding Transformer models)
  - A collection of documents.
  - General
  -
  -
  -
* - BART/T5-like (also called sequence-to-sequence Transformer models)
  -
  -
  -
  -
  -
* - Encoder
  - Encoder models have bi-directional attention. Encoder models are best suited for tasks requiring an understanding of the full sentence, such as sentence classification, named entity recognition (and more generally word classification), and extractive question answering.

    More or less, encoder is like a feature extractor, one that extracts **representations** of the input sentence. The representations are then used by the downstream task.
  -
  - One of the key is bi-directional, since it allows the model to understand the context of the word in the sentence. For example, the word bank in "I went to the **bank** to withdraw money"
  will have a different meaning than the word bank in "I went to the **bank** of the river to fish".
  - - [HuggingFace Course: Encoder Models](https://huggingface.co/course/chapter1/5?fw=pt)
  - **Tasks**: Sentence classification, named entity recognition, extractive question answering.

      **Models**: ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa, etc
* - Decoder (Auto-Regressive)
  - Decoder models have uni-directional attention. At each stage, the attention
  layer is only allowed to look at the previous tokens in the sequence.
  -
  -
  - - [HuggingFace Course: Decoder Models](https://huggingface.co/course/chapter1/6?fw=pt)
  -
* - Encoder-Decoder (Sequence-to-Sequence)
  -
  -
  -
  - - [HuggingFace Course: Encoder-Decoder Models](https://huggingface.co/course/chapter1/7?fw=pt)
  -
```
````

