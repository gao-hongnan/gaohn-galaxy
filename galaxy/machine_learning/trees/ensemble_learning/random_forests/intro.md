# Random Forests

Random forest is an ensemble learning method that is used for classification and regression problems. It combines the results of multiple decision trees to produce a more robust and accurate prediction. In a random forest model, multiple decision trees are trained on different subsets of the data and features. The final prediction is made by combining the predictions of all the trees.

In a random forest, each decision tree is trained on a randomly selected subset of the training data, and a randomly selected subset of the features. This helps to reduce the variance of the model, as well as to decorrelate the trees. The final prediction is made by aggregating the predictions of the individual trees, typically by taking the average or majority vote. The random selection of data and features also helps to prevent overfitting, which is a common issue with single decision trees.

Random forest models are fast, easy to use, and produce good results in many practical applications. They are also relatively robust to noise in the data and to missing values, and they provide a feature importances score which can be used to select the most important features.

## Further Readings

- Vincent Tan notes
- Murphy, Kevin P. "Chapter ." In Probabilistic Machine Learning: An Introduction. MIT Press, 2022.
- James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. "Chapter ." In An Introduction to Statistical Learning: With Applications in R. Boston: Springer, 2022.
- Jung, Alexander. "Chapter " In Machine Learning: The Basics. Singapore: Springer Nature Singapore, 2023.
- Bishop, Christopher M. "Chapter ." In Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2016.
- Hal Daumé III. "Chapter ." In A Course in Machine Learning, January 2017.
- [Machine Learning from Scratch](https://dafriedman97.github.io/mlbook/content/introduction.html)
- **GOOD**: https://github.com/NathanielDake/intuitiveml
- https://github.com/goodboychan/goodboychan.github.io/tree/main/_notebooks