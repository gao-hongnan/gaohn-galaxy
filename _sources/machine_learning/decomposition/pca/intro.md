# Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to reduce the complexity of a high-dimensional dataset by finding a lower-dimensional representation of the data. It does this by identifying the principal components of the data, which are the directions along which the data varies the most. The first principal component is the direction that captures the largest amount of variance in the data, and each subsequent component captures less and less variance. By selecting the most important components, PCA can be used to simplify the data and make it easier to visualize and analyze. Additionally, it can be used for data compression, noise reduction, and to uncover hidden patterns or relationships in the data.

## Further Readings

I think the MML book has a nice treatment of PCA, so do check it out first.

- Jung, Alexander. "Chapter 9.2. Principal Component Analysis." In Machine Learning: The Basics. Springer Nature Singapore, 2023.
- Deisenroth, Marc Peter, Cheng Soon Ong, and Aldo A. Faisal. "Chapter 10. Dimensionality Reduction with Principal Component Analysis." In Mathematics for Machine Learning. Cambridge: Cambridge University Press, 2021.

### Common FAQ

- [Do you need to center the data before applying PCA?](https://stats.stackexchange.com/questions/189822/how-does-centering-make-a-difference-in-pca-for-svd-and-eigen-decomposition)