
# COVID-19 Inpatient Data Analysis and Machine Learning

**Author:** Aditya More  
**Program:** MS in Data Science, University at Buffalo  
**Course:** MGS 670  
**Project Type:** Healthcare Analytics | Machine Learning | Deep Learning

## 1. Project Overview

Hospital length of stay (HLOS) is an important outcome in healthcare operations. Understanding factors associated with longer hospitalization can support research into patient outcomes and hospital resource planning.

This repository contains two related machine learning projects examining COVID-19 inpatient data.

- **Project 1:** Clinical data preprocessing, exploratory analysis, Multilayer Perceptron (MLP) regression, and Recurrent Neural Networks (RNNs).
- **Project 2:** Admission-based prediction using multi-head self-attention, Word2Vec representations, and MLP baselines.

The projects demonstrate an exploration of traditional and advanced neural network approaches for hospital length-of-stay prediction.

## 2. Research Objectives

The main objectives were to:

1. Examine clinical characteristics of hospitalized COVID-19 patients.
2. Identify and handle missing or inconsistent medical measurements.
3. Prepare admission and longitudinal clinical data for machine learning.
4. Develop and compare neural network regression architectures.
5. Evaluate predictive performance using MSE and RMSE.
6. Investigate the effects of model architecture, hyperparameters, and feature representations.
7. Understand limitations associated with predicting clinical outcomes from a relatively small patient cohort.

## 3. Dataset Description

The analysis uses the `Canada_Hosp1_COVID_InpatientData.xlsx` dataset.

### Dataset Summary

| Characteristic | Value |
|---|---|
| Total patients | 508 |
| Admission records | 508 |
| Longitudinal daily records | 4,064 |
| Admission worksheet columns | 55 |
| Median hospital length of stay | 8 days |
| Mean hospital length of stay | 12.43 days |
| Minimum hospital length of stay | 1 day |
| Maximum hospital length of stay | 81 days |

### Dataset Worksheets

The Excel workbook contains four worksheets:

- `Data-at-admission`
- `Days-breakdown`
- `Hospital-length-of-stay`
- `Medication-Static-List`

### Clinical Variables

The dataset includes several categories of clinical information:

**Patient characteristics**
- Age and sex
- Height and weight
- Medical comorbidities

**Vital signs**
- Heart rate
- Respiratory rate
- Blood pressure
- Oxygen saturation
- Body temperature

**Laboratory measurements**
- White blood cell count
- Hemoglobin
- Serum creatinine
- Sodium and potassium
- Liver function measurements
- Inflammatory and coagulation markers

**Clinical outcomes**
- Hospital length of stay
- ICU-related outcomes
- Mechanical ventilation information
- Mortality-related outcomes

Not all outcome variables were used as predictors.

## 4. Project 1: Clinical Data Analysis and Neural Networks

### 4.1 Data Preprocessing

The preprocessing workflow included:

- Examining dataset structure and missing values.
- Converting numeric fields stored as text.
- Encoding comorbidity information.
- Removing selected features with excessive missingness.
- Applying missing-value imputation.
- Standardizing numerical predictors.
- Preparing patient-level training and testing datasets.

The MLP analysis used 53 predictors after preprocessing.

### 4.2 Exploratory Data Analysis

Descriptive statistics were used to investigate patient characteristics, clinical measurements, and hospital length-of-stay distributions.

An interquartile range (IQR) analysis was also performed to identify potentially extreme observations.

Outliers were not automatically removed because unusual clinical measurements can reflect genuine illness severity rather than incorrect data.

### 4.3 Multilayer Perceptron (MLP)

Several MLP configurations were evaluated to investigate the effects of network size, regularization, and learning rate.

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| Baseline MLP | 203.991 | 14.283 |
| Smaller MLP | 227.852 | 15.095 |
| Regularized MLP | 210.605 | 14.512 |
| Lower Learning Rate MLP | 151.911 | 12.325 |

**Finding:** The lower-learning-rate MLP achieved the lowest test error among the four configurations in this experiment.

### 4.4 Recurrent Neural Network (RNN)

RNN models were investigated using longitudinal patient observations.

The sequence dataset contained:

- 508 patients
- 8 observation time points per patient
- 25 predictor features
- Training data shape: (406, 8, 25)
- Testing data shape: (102, 8, 25)

The RNN experiments modeled remaining hospital length of stay over the observation sequence.

| Model | Test MSE | Test RMSE |
|---|---:|---:|
| Baseline RNN (32 units) | 124.127 | 11.141 |
| Smaller RNN (16 units) | 117.700 | 10.849 |

**Finding:** The smaller RNN achieved the lowest test MSE among the two reported configurations.

The RNN and MLP experiments use different target structures, so these results are not a controlled comparison of identical prediction tasks.

### 4.5 Five-Trial Evaluation

Additional experiments used five training trials to examine model performance variability.

| Model | Mean Test MSE | Test MSE Std. Dev. |
|---|---:|---:|
| MLP | 246.683 | 61.064 |
| RNN | 115.993 | 17.964 |

The repeated experiments revealed variation across runs, highlighting the importance of evaluating model stability rather than relying on a single training result.

## 5. Project 2: Attention and Word2Vec

Project 2 investigated more advanced neural network approaches using admission-based clinical information.

The prediction target was total hospital length of stay.

### 5.1 Feature Selection and Leakage Prevention

The project used 508 patient records, with 33 numerical and 4 categorical admission features identified during feature selection.

The preprocessing approach included:

- Numerical imputation
- Feature standardization
- Categorical encoding
- Training-fold-only preprocessing
- Patient-level data splitting

Post-admission outcomes such as ICU length of stay, ventilation duration, and mortality-related information were excluded from admission-based prediction inputs to reduce outcome leakage.

### 5.2 Model Architectures

Three modeling approaches were compared.

**A. Multi-Head Self-Attention**

A PyTorch attention-based model was developed using:

- Learnable feature representations
- Multi-head self-attention
- Pooled feature representations
- A regression prediction head

**B. MLP Baseline**

A feedforward neural network was used as a baseline for comparison.

**C. Word2Vec + MLP**

Word2Vec representations were incorporated into an MLP-based prediction approach to investigate whether learned feature representations improved predictive performance.

### 5.3 Model Evaluation

Five patient-level folds were used for evaluation.

Hyperparameter configurations were selected using training data within the evaluation procedure.

Model performance was assessed using mean squared error (MSE) and root mean squared error (RMSE).

### 5.4 Five-Fold Results

| Model | Mean Test MSE | Test MSE Std. Dev. | Mean Test RMSE |
|---|---:|---:|---:|
| Attention | 185.906 | 41.581 | 13.574 |
| Word2Vec + MLP | 191.022 | 49.234 | 13.734 |
| MLP Baseline | 228.503 | 55.451 | 15.025 |

**Key observations:**

- The attention model achieved the lowest mean test MSE across the five folds.
- Word2Vec + MLP achieved approximately 16.4% lower mean test MSE than the MLP baseline.
- Performance varied across folds.
- The results suggest that learned feature representations merit further investigation.

The reported results do not establish statistically significant superiority or clinical readiness.

## 6. Technologies and Libraries

| Category | Technologies |
|---|---|
| Programming | Python |
| Development environment | Jupyter Notebook |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow, Keras, PyTorch |
| Feature representations | Gensim, Word2Vec |
| Visualization | Matplotlib |
| Evaluation | MSE, RMSE, repeated trials, cross-validation |

## 7. Challenges and Technical Decisions

### Missing Clinical Data

Several clinical variables had substantial missingness.

The projects explored missing-value analysis, feature exclusion, and imputation to prepare the dataset for machine learning.

### Outlier Handling

Clinical observations can contain extreme values associated with severe patient conditions.

An IQR-based screening procedure was used, but extreme observations were not automatically discarded.

### Overfitting and Generalization

Differences between training and testing errors demonstrated the need to consider generalization performance.

The experiments investigated different network sizes, regularization settings, and learning rates.

### Evaluation Variability

Repeated trials and cross-validation revealed variation in model performance.

This reinforced the importance of reporting variability alongside average prediction errors.

### Data Leakage

Project 2 applied preprocessing within training folds and restricted predictors to admission-related information.

These decisions were intended to make the evaluation more representative of prediction using information available at admission.

## 8. Key Learning Outcomes

These projects provided practical experience with:

- Working with multi-sheet clinical datasets.
- Cleaning and transforming healthcare data.
- Handling missing measurements and outliers.
- Engineering machine learning features.
- Implementing MLP and RNN models.
- Developing attention-based neural network architectures.
- Exploring Word2Vec representations.
- Comparing models using repeated experiments and cross-validation.
- Interpreting model performance and generalization limitations.
- Communicating technical findings through reproducible notebooks.

## 9. Repository Files

- `MGS670_Project1_AdityaMore_FINAL.ipynb` — Clinical preprocessing, MLP, RNN, and evaluation experiments.
- `MGS670_Project2_AdityaMore_.ipynb` — Attention, Word2Vec, MLP baselines, and five-fold evaluation.
- `README.md` — Project documentation.
- `.gitignore` — Git exclusion rules.

## 10. How to Run the Projects

1. Clone or download this repository.
2. Set up a Python environment with Jupyter Notebook.
3. Install the libraries imported by the relevant notebook.
4. Obtain authorized access to the original Excel dataset.
5. Open the notebook you want to run.
6. Follow its dataset-loading instructions and update file paths where necessary.
7. Run the notebook cells in order.
8. Review the model evaluation outputs and visualizations.

The original dataset is not included in this repository. Project 1 uses an interactive upload step, while Project 2 requires the expected workbook to be available at its configured location.

The notebooks may require environment-specific adjustments to reproduce the saved results.

## 11. Limitations and Future Improvements

Several limitations should be considered:

- The dataset contains only 508 patients.
- Some clinical measurements have substantial missingness.
- Model performance varies across training trials and evaluation folds.
- Results have not been externally validated on an independent hospital cohort.
- Different prediction setups prevent direct comparison of every model across the two projects.
- These experiments are educational and exploratory, not clinical decision-support systems.

Potential future improvements include external validation, additional baseline models, model explainability analysis, and more extensive hyperparameter optimization.

## 12. Author

**Aditya More**  
Master of Science in Data Science  
University at Buffalo, State University of New York

[GitHub](https://github.com/Adi0528)

[LinkedIn](https://www.linkedin.com/in/aditya-p-more-78a054285/)

---

*This repository documents completed academic machine learning experiments for educational and research portfolio purposes. It does not provide medical advice or validated clinical predictions.*

