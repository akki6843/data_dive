# Understanding the intuition for Random forest :  
    Random forest is a type of ensemble Learning.

    Step 1 : Pick at random K data point from the training set.  
    Step 2 : Build the decision tree on this K data points.  
    Step 3 : Choose the number of Trees N you want to build and then repeat this steps.  
    Step 4 : For predicting new data point use all the N tress to predict the values of Y for the data point in Question. Now assign the average Y predicted from all the trees that predicted Y.  

So it will be like using lots of trees life forest of trees for predictions, Hence Random Forest.