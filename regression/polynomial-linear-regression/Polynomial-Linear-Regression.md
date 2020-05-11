# Understanding What is polynomial linear regression  

Y = b0 + b1*x1 + b2*x1^2 + b3*x1^3 + ... + bn*x1^n  

Here linear is because, of coefficients. And the point here is can we express the function using coefficient.(This is a special case of Multi-linear regression)  

# Along with scikit-learn library, concept is also attempted in tensorflow.  
    ## Observation during tensorflow implementation :  

        - Amount of data required for neural network based approach is high.
        - Tensorflow model gives nothing if training data is very small. Vanishing gradient has been observed.
        - Will re-attempt this after finding better dataset.

