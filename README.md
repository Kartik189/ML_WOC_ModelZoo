# ML_WOC_ModelZoo

a.) Using Machine Learning to predict CO2 emissions from a Vehicle by using a dataset provided.
It involves using LinearRegression Algorithm to do so.
1. Uploading Data And Filtering it(Correcting DataTypes,Nan Values etc if present)
2. Normalizing our data
3. Distributing data into Training And Testing Set.
4. Training Our Model.
5. Using it to test/predict data.
6. Using variance score to test the efficiency of our model

Also, after using polynomial regression via pipelines we get a score R^2 score of 0.91 which is much better than what we got from our Linear Regression Model


b.)Using Machine Learning to predict stability of a star system based on a few parameters.
1.Using LogisticRegression function in scikit-learn(But because there was no attribute to vary alpha i choose 3.svm classifier to vary c and minimize the error) .
2.Developing logistic regression alogorithm on my own .
3.Using SVM Classifier and varying the values of c and max_iter the mis-classification error(=0.05931) was minimized for c=25.0 and max_iter=2500.0 
