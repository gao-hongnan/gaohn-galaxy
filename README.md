<div align="center">
<h1>Machine Learning Chronicles</a></h1>
by Hongnan Gao
March, 2023
<br>
</div>


<h4 align="center">
  <a href="https://gao-hongnan.github.io/gaohn-galaxy/intro.html">Documentation</a>
</h4>


## Introduction

This is the documentation for the course [Introduction to Probability for Data Science](https://probability4datascience.com/).

The bottom-up learning takes place in 3 phases:

- **Conceptual Understanding**, where you learn the intuition, concepts and the mathematics behind the algorithms.
More often than not, you might get stuck in this phase. It is important to keep going and not give up. Things
may be clearer later on.
- **Implementation**, where you implement the algorithms from scratch. This is the phase where you get to know
the inner workings of the algorithms. Unfortunately, while knowing how to ***implement*** does not
necessarily mean your ***understanding*** is deep, it is still a good way to learn and build towards
a deeper understanding. It is particularly useful when your models spectacularly fail one day, and
you know exactly where to look for the bug.
- **Application**, where you apply the algorithms to real-world problems. This is the phase where you
try to connect the dots and see the big picture.

At some stage, it is good to know the **motivation** of certain concepts. These will be included
in the introduction of each chapter.

**Note that the implementation phase may not always be feasible.**

## Workflow

### Installation

```bash
~/gaohn                     $ git clone https://github.com/gao-hongnan/gaohn-galaxy.git gaohn-galaxy
~/gaohn                     $ cd gaohn-galaxy
~/gaohn/gaohn-galaxy        $ python -m venv <venv_name> && <venv_name>\Scripts\activate
~/gaohn/gaohn-galaxy (venv) $ python -m pip install --upgrade pip setuptools wheel
~/gaohn/gaohn-galaxy (venv) $ pip install -r requirements.txt
~/gaohn/gaohn-galaxy (venv) $ pip install -r requirements_jupyterbook.txt
```


### Building the book

After cloning, you can edit the books source files located in the `galaxy/` directory.

You run

```bash
~/gaohn/gaohn-galaxy (venv) $ jupyter-book build galaxy/
```

to build the book, and

```bash
~/gaohn/gaohn-galaxy (venv) $ jupyter-book clean galaxy/
```

to clean the build files.

A fully-rendered HTML version of the book will be built in `galaxy/_build/html/`.