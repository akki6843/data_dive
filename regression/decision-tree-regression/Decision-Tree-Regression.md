# Understanding Decision Tree Regression :  
    CART : Classification and Regression Trees  

This topic deals with regression tree.    
In Decision Tree the data points are split into multiple zones. This zones are then further used to provide the prediction of dependent variable.  

So the Question is how exactly the prediction happens.  
When the training data is provided to decision tree regression, Algorithm splits the data into multiple zones based on the tuning parameters. And assign them a average value.  
At the time of prediction when a new data-point is provided it is traversed through the decision tree like a nested if statement. And wherever conditions are satisfied, Algorithm returns the average value as a prediction.  


# Note : Decision  tree do not adapt good for less amount or one feature. Also Decision Tree Regression does not require feature scaling   

Decision Tree Regression works on continues values. And Decision Tree Classifier works on categorical data.