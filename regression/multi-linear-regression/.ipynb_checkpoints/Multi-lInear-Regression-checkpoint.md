# Understanding What is Multi-Linear-Regression  

Y = b0 + b1*x1 + b2*x2 + ... + bn*xn

### Linear Regressions have assumptions  
    - Linearity 
    - Homoscedasticity
    - Multivariate Normality
    - Independane of errors
    - Lack of Multicollinearity  
Identify what are this assumptions.  

### When a dataset have multiple independent Variables and one dependent variable.  
#### When a dataset has categorical data then, expand the dataset and use dummy variables for fillting them in one regression model.
### What is Dummy Variable Trap ?  


### p-value:  It is measure of strength of the evidance against the null hypothesis.

# Building Model Methods:
## All in method :
    * its not a very reliable meathod,as it can cause the model to genrate baised outpu
    * This method of createing the model is used only if :
        - all the input columns are very much known.
        - the source of the data is already cureated and trusted.
        - :; if you have no freedom of filtering the dataset.
        
## Backward Elimination: (Fastest one)
    Step 1: Select the significance level  to stay in the model (0.05 is most comanlly used value).
    Step 2: Fit full model with all the possible predictors 
    Step 3: Consider the predictor ith the higest P-Value. If P > SL(significance level)
            Step-4: Remove the predictor 
            Step 5: Fit model without thi variable.
         Else:
            Model is ready

## Forward Selection :
    Step 1: Select the Signifcance level to enter thr model (e.g. SL = 0.05 or 05%) 
    Step 2: Fit all the simple regression model y = m*xn Select the Independent variable with the lowest P-value.
    Step 3: Keep this variable and fit all the possible models with one extra predictor added to the one we already have.
    Step 4: Consider the predictor with the lowest P-Value. If P < SL
                                                                Step 3
                                                            else
                                                                Model is ready
## Bidirectional Elimination:
    Step 1 : Select the Significance Value to enter and stay in the model.
    Step 2 : Perform the steps of forward Selection.
    Step 3 : Perform the steps of Backward-Elimination.
    Step 4 : NO new variable can enter and no old variable can exit.
            Model is ready
## All possible Models:
    - its most through model and very resource hungry method.
    Step 1 : Select the criteria of goodness of fit.
    Step 2 : Construct all possile Regression Models: (2^n - 1) total combinations.
    Step 3 : Slect the one with the best criterion.
        MOdel is ready.
 