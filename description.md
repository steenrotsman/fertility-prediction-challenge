# Modeling choices
We worked with a 50-50 training and validation split to get a sense of which models and features were working well and which were not. The splits were randomly generated at every iteration, allowing us to monitor variation in performance due to different training data. The splits were stratified on the outcome variable and made so that members of the same household would occur in the same split. 

LGBM offers a fast and accurate model for classification which is able to handle missing data in tree splitting. We tested other models, including Logistic Regression, Random Forests, Naive Bayes, and converged on LGBM for the final submission due to a robustly better performance. 

We tweaked the decision boundary, to accommodate for the highly imbalanced problem and found that predicting someone will have a child if the model assigns a probability higher than 28%. 

# Missing data imputation
We considered running MICE (while keeping track of which responses were not provided, after observing that the mere presence/absence of a response on our set of features was rather predictive, suggesting that non-responses were informative), but decided against it because it was not clear whether imputation on the test set could have relied on information in the training set. Moreover, since LGBM handles missing data without the need to impute, we decided against MI. 

# Feature selection

We started from the full set of features in background and core studies, adopting a full bottom-up approach. For every individual, we added the last available value of the background characteristics and dropped variables with textual, date or time response types. 

We then inspected confusion matrices and feature importance to gauge which waves and features provided the most information. This process highlighted the need to fine-tune the decision boundary to avoid a high number of false negatives. Moreover, we observed that important features consistently concerned waves from 2017 on. We thus discarded all previous waves to have a more manageable set of features. Feature importance confirmed that the model relied on features related to fertility, such as questions asking about fertility intentions.  

# Feature engineering
From the variables that were consistently found important for different data splits, we derived new features. First, to reduce the missingness rate, we combined the most recent non-missing answers from the waves to questions relating to fertility intentions (cfxxx029, cfxxx128, and cfxxx130), as well as those about key family events, such as getting married (cfxxx031) and becoming parents (cfxxx456). 

The background features were consistently considered important as well. Due to the low missingness of these variables, the prior strategy was unnecessary. Instead, we recorded the number of unique values across the years for variables related to housing situation (woning, sted, woonvorm, partner), education (oplzon, oplmet), and the variables burgstat and belbezig. The number of unique values over the years for a person can indicate changes in these variables. Another change in response values that we extracted was that of net household income (nettohh_f). For this variable, we calculated the slope for each individual by fitting a linear regression to the yearly observations. 

# Observations

We did observe that fertility intentions do predict fertility outcomes. Moreover, we observed household income to matter more than individual income. Marital status also consistently showed up as an important feature. 

# Outlook with more data or registry data

More feature engineering, concerning for example neighborhood statistics and social networks, is likely to improve performance. However, if we are given the opportunity to work with CBS microdata, our approach would likely change and veer towards graph networks that exploit the rich network structure available in microdata.  