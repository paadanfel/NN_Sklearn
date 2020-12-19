# Reflection

The California Housing Dataset was used to perform the tasks.

[California Housing Dataset](`https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html (Links to an external site’))

### Predicting the value of homes

**Procedure**

To predict the value of homes, the MLP Regressor from Sci-kit learn was used.

Before we began the analysis, the MedHouseVal column was scaled with the standard scaler, and the entire data was split 80-20 into training and testing.

To properly get the analyzed data, we looped over the different activations, the number of hidden layers, alpha values, and learning rates.

This generated 384 different models.

The Adam was used as a solver, as it was able to manage larger datasets compared to `lbfgs` or `sgd` since it was a definite possibility to increase the maximum iteration to obtain similar results.

When comparing each model, we used $R^2$, Train Loss, Test MSE, and Time Taken. The result was saved into the CSV file, as the execution took at least 14 minutes, even with the TPU from the Google Colab.

**Result**

By plotting the data Activation vs. Time taken, regardless of the Alpha value, the logistic activation took the longest time.

The learning rate of 0.1 took the shortest time taken.

When it comes to the activation functions, identity and relu had way more errors (inferring from Train Loss and Test MSE) to compare logistic and tanh.

From these two activations, tanh produced less errors in both hidden units 1 and 2.

When comparing two activations on Test MSE, $R^2$, Train Loss, and Time Taken, it was clear that tanh with hidden units of 5 and 5 produced the best performance.

Overall, taken into consideration the metrics above, the best combination of the 384 is 
```python 
MLPRegressor(hidden_layer_sizes=(5, 5), activation='relu', solver='adam', alpha=0.001, learning_rate_init=0.1)
```


### Predicting the category of home values

**Procedure**

To predict the category of home values (with values greater than 2 as 1 and 0 otherwise), we performed a similar analysis in terms of the for loop with the only difference being the MLP Classifier.

This also resulted in 384 models.

We fitted the training data with the MLP Classifier and continued predictions with the testing data.

For each model, we compared the Accuracy, Precision, Recall, and F1 score.

We similarly plotted the data regarding hidden units vs. time taken; however, we created four plots next to each other that resembled the relationship among hidden units given their learning rate, activation, and alpha.

**Result**

We started with a couple of plots that placed activations against the time taken given alpha and learning rates.

From the plots, we obtained that both alphas increase linearly from identity to logistic; however, the alpha of 0.0001 significantly decreases.

The learning rates in the plot scatter; however, a learning rate of 0.1 is very consistent in the less time taken.

We used an intersection among vertical and horizontal lines that displayed the maximum value among accuracy, f1_value, precision, and recall.

Through these intersection, we concluded that activation of identity and hidden units of 20 and 5 produced the best performance i.e 
```python
MLPClassifier(hidden_layer_sizes=(20, 5), activation='identity', solver='adam', alpha=0.0001, learning_rate_init=0.1)
```

## License

The license to be used is [MIT](https://choosealicense.com/licenses/mit/)