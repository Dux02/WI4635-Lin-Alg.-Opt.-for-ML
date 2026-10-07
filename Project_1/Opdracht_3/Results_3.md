## Results 3
Double descent is a phenomenon where the test error first decreases as model complexity
increases, then increases sharply near the interpolation threshold (where the model is just able
to perfectly fit the training data), and then decreases again as model complexity increases
further. For neural networks, the phenomenon is not yet fully understood.
For OLS, we can investigate double descent by varying the number of data points while keeping
the number of features fixed. (This is called sample-wise double descent.) Construct synthetic
data to investigate this phenomenon, and use the minimum-norm OLS solution when there are
fewer data points than features. Plot the training and test errors as functions of the number
of training data points. Which properties of your dataset influence the phenomenon? Explain
the behaviour you observe.