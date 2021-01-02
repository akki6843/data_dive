# R-Square and R-Square Adjusted Intuition

## R-Square is a parameter that help in identifying how good the line is fitting compared to the average. So in simple terms R-Squared tells how well the model is fitted to the data

<p>&nbsp;</p>

## R-Squared is Calculated as :

    Rsq = 1 - ( SSres / SStot )  
    Where,   
        SSres = Squared SUM of ( Yi - Yhat )   
        SStot = Squared SUM of ( Yi - Yavg )  
            here,  
                Yi   =  Actual value or label value
                Yhat =  Predicted Value
                Yavg =  Average value of Yi

## R-Squared value 1 is the ideal scenario. So the closer the value of R-Squared to 1 , better the model on test scenario. And it is possible that R-Squared value goes in negative but its very difficult to produce.  

<p>&nbsp;</p>

## Problem with R-Squared is that if we increase the number of dependent variables to improve the model efficiency its hard to detect in R-Squared parameter for the impact made by adding the new variable  
  
<p>&nbsp;</p>

## So to address this problem R-Squared is further tricked in to Adjusted R-Squared.  

<p>&nbsp;</p>

# Adjusted R-Square  : 
## Adjusted R-Squared helps in penalizing the dependent variables that are not contributing for the model convergence.
<p>&nbsp;</p>

## Adjusted R-Squared is calculated as :

    Adjusted R-Square = 1 - (1 - R-Squared)(n-1/n-p-1)
    Where,
        p = number of regressors 
        n = sample size.