# Deep Learning Notations

We largely follow the [Machine Learning: The Basics](https://link.springer.com/book/10.1007/978-981-16-8193-6)
book in terms of notations.

## NLP Notations

Given a dataframe of five rows:

```
1. This is Foo, how can I help you
2. It might rain today
3. I love football
4. Crazy, Stupid & Love
5. I shot the sheriff
```
https://stackoverflow.com/questions/72550835/corpus-vs-vocabulary-vs-document-in-nlp
We will use this table to explain some terminologies below.

In NLP, document concept can be a bit vague: a document is a unit, so it can correspond to different text objects, such as entire documents, sentences, passages... In your example, "This is Foo, how can I help you" is a document. "It might rain today" is another document...

A corpus is a collection of documents. In your example, the corpus is composed by 5 documents.

The vocabulary is the list of all the words contained in the corpus, therefore all the words contained in all the documents. Your vocabulary is [&, can, crazy, foo, football, help, how, i, is, it, love, might, rain, sheriff, shot, stupid, the, this, today, you]

```{list-table} Basic NLP Terminologies
:header-rows: 1
:name: basic-nlp-terminologies

* - Notation
  - Description
* - $V$
  - The dimension of the number of vocabulary words in the corpus.
* - Corpus/Collection
  - A collection of documents.
```