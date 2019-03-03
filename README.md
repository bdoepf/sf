# SF 
Playground for Spark ML on a kaggle dataset [San Francisco Crime](https://www.kaggle.com/c/sf-crime).
It tries to predict the crime category. Therefore it's a classification problem.

The main app uses Spark ML Pipelines with several different ML algorithms at the same time. It uses cross validation for hyper parameter tuning and model selection.

# Using Auto ML on SparkML
The `com.github.bdoepf.sf.auto` package is a playground for trying Saleforce's [transmogrifai auto ML lib](https://github.com/salesforce/TransmogrifAI) which is build upon SparkML.