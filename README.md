# Binary-Classification-of-Cardiovascular-Disease-and-Cancer-cells-Using-AWS-AutoGluon
Auto ML Classification


This project aims to train several machine learning classifiers to detect and classify cardiovascular disease and cancer on autopilot using the AutoGluon Machine Learning library.
AutoGluon is the library behind Amazon Web Services (AWS) autopilot and it allows for quick prototyping of AI/ML models using a few lines of code.

### Practical application of this project:
- Cardiovascular disease (CVD) remains as the leading cause of death in the United States, accounting for 928,741 deaths in the year 2020 (1).
- Therefore, the steps and results of this project may be useful to physicians or other healthcare professionals to help them realize how ML models can help detect cardiovascular disease, and better understand the confounding factors that contribute to the disease.

Data Variables:
1. Inputs:
* Age, Height, Weight, Gender
* Smoking, Alcohol intake, Physical activity
* Systolic blood pressure, Diastolic blood pressure
* Cholesterol, Glucose

2. Output:
* Cardiovascular disease (1 or 0) #binary classification output

*Data Source*
- https://www.kaggle.com/sulianova/cardiovascular-disease-dataset

<p>&nbsp;</p>

Correlation matrix relates all the features.
- There is a strong positive correlation of 0.45 between glucose and cholesterol levels.
- There is also a strong positive correlation of 0.50 between gender and height.

  ![image](https://github.com/IkChristine/Binary-Classification-of-Cardiovascular-Disease-and-Cancer-Using-AWS-AutoGluon/assets/104997783/e66d7948-9d30-466a-80cf-77030a104c49)



### AutoGluon is modularixed into sub-modules and the one used in this analysis was 
- autogluon.tabular - tabular data (TabularPredictor)
  
### AutoGluon Presets understanding: Presets include:
- Best quality: Best predictive accuracy with little consideration to inference time or disk usage.
- High quality fast inference only: High predictive accuracy with fast inference and lower disk usage than best quality. 
- Medium quality faster train: (Default Preset) Medium predictive accuracy with fast inference and very fast training time.
- Optimize for deployment: optimizes results for deployment by deleting unused models and removing training artifacts. Can reduce disk usage without impacting model accuracy or inference speed.


#### Training multiple ML regression models using AutoGluon
- Specify the target column 'cardio', train_data, limit_time, and presets 
- AutoGluon automatically detects if the problem is classification or regression type problems from the 'label' column

predictor = TabularPredictor(label = "cardio", problem_type = 'binary', eval_metric = 'accuracy').fit(train_data = X_train, time_limit = 200, presets = "medium_quality_faster_train")


![image](https://github.com/IkChristine/Binary-Classification-of-Cardiovascular-Disease-and-Cancer-Using-AWS-AutoGluon/assets/104997783/52e15814-d0a5-4e28-96fb-63679da58415)

- Best performing model is the **WeightedEnsemble_L2 with about 74% accuracy**
- least performing model is the  KNeighborsDist with about 67% accuracy

  
![image](https://github.com/IkChristine/Binary-Classification-of-Cardiovascular-Disease-and-Cancer-Using-AWS-AutoGluon/assets/104997783/fddc17ec-68ca-46e1-8190-e5439fde8529)


If model prediction (y_pred) matches ground truth data (y_test) then the model performed well.

- The model correctly classified 5544 as not having cardio vasular disease (True Negative)
- The model correctly classified 4769 as having cardiovascular disease (True Positive)

- The model misclassified 1525 and not having cardiovascular disease but they actually had the disease (False Negative)
- The model misclassified 2162 has having cardiovascular disease but they did not have the disease in reality (False Positive)


### Cancer dataset -  Using ‘best_quality’ preset and accuracy metric

predictor = TabularPredictor(label = "target", problem_type = 'binary', eval_metric = 'accuracy').fit(train_data = X_train, time_limit = 200, presets = "best_quality")


<img width="775" alt="image" src="https://github.com/IkChristine/Binary-Classification-of-Cardiovascular-Disease-and-Cancer-Using-AWS-AutoGluon/assets/104997783/3a9578e5-7a23-4201-ad0c-4f176c7a34d2">


- The best performing model was WeightedEnsemble_L2 with a validity score of 98%
- The worst performing model was KNeighborsDist_BAG_L1 with a validity score of 92%
- Model performed extremely well with accuracy score of 97%.


References: 
1. American Heart Association. (2023). Heart Disease and Stroke Statistics—2023 Update: A Report From the American Heart Association. Circulation, 147(10), e305–e339. https://doi.org/10.1161/CIR.0000000000001123 
