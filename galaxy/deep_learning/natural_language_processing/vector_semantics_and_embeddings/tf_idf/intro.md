# Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a popular technique used
in information retrieval and natural language processing to quantify the importance
of each word in a document or corpus of documents.

## Motivation

Consider the example in [the section on representing words as vectors using document dimension](words-as-vectors-document-dimensions).
Let's add one more vocab to the matrix, the most frequent word **the**, which appears many times across all documents.

```{list-table} Number of occurrences of selected words (modified) in four Shakespeare plays.
:header-rows: 1
:name: term-document-table-modified

* -
  - As You Like It
  - Twelfth Night
  - Julius Caesar
  - Henry V
* - battle
  - 1
  - 0
  - 7
  - 13
* - good
  - 114
  - 80
  - 62
  - 89
* - fool
  - 36
  - 58
  - 1
  - 4
* - wit
  - 20
  - 15
  - 2
  - 3
* - the
  - 1000
  - 1200
  - 900
  - 1100
```

Then by the cosine similarity metric, we will see that many words will be similar to the word `the`.
This is skewed information, if you consider the table above and see the word `the` and `good`, then
if you want to do document retrieval based on these two words alone (hypothetically), will yield you
the document that contains the word `the` the most, which is not what we want. This kind of violate the idea
"similar documents tend to have similar words" mentioned in [words and vectors](../words_and_vectors/concept.ipynb), because the word `the` is so common that it appears in
almost all documents, so it is not a good indicator of similarity.

In what follows, the basic idea behind TF-IDF is to give a high weight to words that appear frequently in a document but rarely in other documents, while giving a low weight to words that are common across all documents. The weight assigned to each word is calculated based on two factors: term frequency (TF) and inverse document frequency (IDF).
This is a solution to the aforementioned problem.


## References and Further Readings

- Jurafsky, Dan, and James H. Martin. "Chapter 6.5. TF-IDF: Weighting terms in the vector." In Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. Pearson, 2022.