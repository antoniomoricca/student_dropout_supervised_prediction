# Student Dropout Detection - Supervised Learning Approach
Applying supervised learning to predict student dropout using neural network and XGBoost. It is a binary classification problem with an unbalanced dataset (class 0 --> dropout students, underrepresented, 15% of the total).

## Introduction

Student dropout is a critical issue for educational institutions, impacting financial sustainability, institutional credibility, and student outcomes. Early identification of at-risk students is essential to enable timely and effective interventions.

This project explores the use of supervised machine learning techniques—specifically XGBoost and Neural Networks—to predict student dropout. A unique aspect of the analysis is its multi-stage design: the same predictive modeling process is applied three times, each using a different dataset representing a specific stage in the student journey.

- Stage 1: Data available at enrollment (e.g., applicant and course information)

- Stage 2: Adds engagement data (e.g., attendance records, authorized/unauthorized absences)

- Stage 3: Includes academic performance data (e.g., number of modules attempted, passed, and failed)

The goal is to determine at which stage it becomes possible to reliably identify students at risk of dropping out. By comparing model performance across stages, the aim is to assess the added predictive value of new information and understand when interventions are most feasible and effective.

## Dataset EDA and Feature Engineering

The dataset contains 25059 rows, each representing a specific learner. The dataset come from a real university, the names had to be deleted. The features are the following:

![image](https://github.com/user-attachments/assets/a3e11a6d-7df0-43ca-b01c-1a98511f783e)

Per each stage, the dataset was cleaned, and feature engineering was performed based on the following criteria:
- Feature Removal: Eliminated LearnerCode as it was irrelevant and HomeState due to over 50% missing values.
- High Cardinality Features: Removed categorical features with more than 200 unique values (HomeCity, ProgressionDegree).
- Discount Handling: DiscountType had missing values only for students without discounts. Created a binary feature (HasDiscount) indicating whether a student received any discount or not and removed the original column.
- Feature Grouping: Aggregated CourseName, StudyArea, and ProgressionUniversity into broader categories to reduce the number of one-hot encoded columns.
- Handling Absences: Rows with missing AuthorisedAbsenceCount and UnauthorisedAbsenceCount (less than 1% of the dataset) were dropped.
- Stage 3 Missing Data: The AssessedModule, PassedModules, and FailedModules features had missing values, which further analysis revealed were only for students who had dropped out. Since these students did not attempt any modules, their missing values were replaced with 0. Additionally, a new feature, SuccessRate (PassedModules/AssessedModule), was created, assigning 0 to students who had not attempted any modules.

After that, encoding was performed on the categorical features: ordinal encoding for the CourseLevel feature (order matters), while for the others one-hot encoding was performed, trying to group and reduce the total number of columns.

## XGBoost

For each dataset, an initial XGBoost model was trained using default parameters to establish a baseline performance. Following this, a grid search was conducted to optimize key hyperparameters, ensuring the model was fine-tuned for better predictive performance. The parameters tested are:
- Learning rate: [0.05, 0.03, 0.01]
- Max depht: [4, 6, 8, 10]
- Number of estimators: [100, 200, 300, 400, 600, .., 1400].

The best configuration has been chosen as the one who maximizes the recall of class 0 (dropout students) with the goal of aiming to identify at-risk students as effectively as possible.

For each dataset, a SHAP plot was generated, showing how certain features pushed predictions toward either class 0 (dropout) or class 1 (course completion). This is a key feature of the XGBoost alg, since it is possible to understand the importance of each feature, leading to a better comprehension of the problem.

Below an example:

![image](https://github.com/user-attachments/assets/41b126d4-41b5-4728-a9d7-593bee904426)

## Neural Network

For each dataset, an initial neural network model was trained using default parameters to establish a baseline performance. Following this, a grid search was conducted to optimize key hyperparameters, ensuring the model was fine-tuned for better predictive performance. The parameters range are:
- Learning rate: [0.001, 0.01, 0.1]
- Number of neuron hidden layer 1: [32, 64, 128]
- Number of neuron hidden layer 2: half of the n neuron hid 1 to reduce computation time
- Activation Functions: [Relu, Tanh, Sigmoid]
- Optimizer: [Adam, RMSProp]
- Batch size: 64

From the various network configurations, it was seen that the performances are quite sensitive to changes in hyperparameters. Unlike XGBoost, which showed much greater stability, the results here vary significantly depending on the chosen configuration. For this reason, instead of optimizing solely for recall of class 0, a more balanced approach was taken by considering both the harmonic F1 score across classes and the weighted F1 score which are plotted below:

![image](https://github.com/user-attachments/assets/dcccc86a-8016-4cf1-99c5-5ae5485acdbe)


## Conclusion

Below the confusion matrix of the 2 models optimized at each stage. The confusion matrix comparison shows that in stages 1 and 2, neural networks outperform XGBoost in identifying at-risk students, with a recall of 65% for class 0, compared to XGBoost's 58% and 61%. However, neither model sees significant performance gains between these stages. In stage 3, both models improve dramatically, correctly identifying over 95% of class 0 cases, confirming that academic performance features are key to distinguishing students at risk of dropping out. The performances of class 1 are consistent across the stages.

![image](https://github.com/user-attachments/assets/43b8833d-a372-4f06-9bec-59ce3d08de44)

![image](https://github.com/user-attachments/assets/7b2ffb1e-d8f0-43a7-9bbd-63e27993f5d3)

This trend highlights the difficulty of identifying dropouts early in the course At stage 1, the model struggles to accurately identify students at risk of dropping out, as early-stage indicators provide limited predictive power. However, XGBoost’s feature importance analysis can still highlight key characteristics of students historically prone to dropping out, offering potential insights for early intervention strategies. By stage 2, mid-course indicators improve the model’s ability to detect at-risk students, capturing around 65% of cases. Despite this progress, the recall remains limited, suggesting that many students who eventually drop out do not exhibit strong warning signs early enough for effective intervention. It is only in stage 3 that clear academic patterns emerge, allowing the model to reliably classify at-risk students.







