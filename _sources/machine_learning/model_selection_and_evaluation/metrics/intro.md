# Metrics and Scoring Rules

Metrics and scoring rules are quantitative measures used to evaluate the performance of a machine learning model. They help us to determine how well the model is able to make predictions, compared to the actual ground truth values.

## Metrics Table

A very comprehensive review can be found in torchmetrics: https://torchmetrics.readthedocs.io/en/latest/.

| Problem Type   | Metric                    | Description                                                                                                                                                                   |
| -------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification | Accuracy                  | The proportion of correct predictions made by a model, out of all the predictions.                                                                                            |
| Classification | Precision                 | The proportion of true positive predictions (i.e., the model correctly predicts that a sample belongs to a positive class) out of all positive predictions made by the model. |
| Classification | Recall                    | The proportion of true positive predictions made by the model out of all actual positive samples.                                                                             |
| Classification | F1-Score                  | The harmonic mean of precision and recall, which provides a single value that balances both metrics.                                                                          |
| Classification | AUC-ROC                   | The area under the Receiver Operating Characteristic curve, which measures the ability of a model to distinguish between positive and negative classes.                       |
| Regression     | Mean Squared Error (MSE)  | The mean of the squared differences between the predicted values and the true values.                                                                                         |
| Regression     | Mean Absolute Error (MAE) | The mean of the absolute differences between the predicted values and the true values.                                                                                        |
| Regression     | R-squared                 | The proportion of variance in the dependent variable that is explained by the independent variable.                                                                           |
| Clustering     | Silhouette Score          | The mean similarity between a sample and all other samples in the same cluster, minus the mean similarity between a sample and all other samples in different clusters.       |
| Clustering     | Calinski-Harabasz Index   | A ratio of between-cluster variance to within-cluster variance, used to evaluate the quality of a clustering solution.                                                        |
| Clustering     | Davies-Bouldin Index      | A measure of the average similarity between each cluster and its most similar cluster, used to evaluate the quality of a clustering solution.                                 |
