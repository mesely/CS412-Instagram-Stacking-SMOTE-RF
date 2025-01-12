# CS412-Instagram-Stacking-SMOTE-RF
# Influencer Category Classification
# Data Preprocessing and Feature Extraction 

---

## Data Preprocessing

### Handling Missing Values
The dataset contained missing values in several features:

- **Features with over 80% missing values** (e.g., `business_category_name`) were excluded.
- **Personal data columns** (e.g., `biography`, `external_url`) had missing values replaced with placeholders like `"no biography"` and `"unknown"`.
- **Categorical features** were imputed using their mode to ensure consistency.

| **Feature**              | **Missing Percentage (%)** |
|--------------------------|----------------------------|
| business_category_name   | 90.22                     |
| post_count               | 89.97                     |
| category_enum            | 40.31                     |
| category_name            | 18.79                     |
| external_url             | 16.67                     |
| entities                 | 6.20                      |
| biography                | 6.20                      |
| full_name                | 0.95                      |

---

## Feature Extraction

### Text Features
1. **Posts**: 
   - Processed using the `emoji` library (e.g., 😄 → `:face_with_tears_of_joy:`), cleaned of URLs, special characters, and whitespace.
   - Converted to lowercase, and numbers were stripped.
   - Vectorized using **TF-IDF** with unigrams and bigrams, generating 5000 features.

2. **Biographies**: 
   - Preprocessed similarly to posts.
   - Vectorized separately with TF-IDF to capture unique patterns.
   - Biography vectors were scaled by a factor of 3 due to their higher relevance, improving the F1 score.

### Numerical and Categorical Features
- **Numerical features** (e.g., `follower_count`, `following_count`) were scaled using Min-Max Scaling.
- **Categorical features** (e.g., `is_verified`, `is_business_account`) were one-hot encoded.

### Combined Dataset
Processed features from text, numerical, and categorical columns were combined to create a unified dataset, improving the model’s ability to generalize across diverse data types.

---

## Handling Class Imbalance

### Using SMOTE
The dataset exhibited significant class imbalance:

| **Class**              | **Initial** | **After SMOTE** |
|------------------------|-------------|-----------------|
| art                   | 200         | 1200            |
| entertainment          | 400         | 1800            |
| fashion               | 150         | 1200            |
| food                  | 120         | 1000            |
| gaming                | 80          | 700             |
| health and lifestyle  | 400         | 1800            |
| mom and children      | 70          | 700             |
| sports                | 100         | 700             |
| tech                  | 200         | 1200            |
| travel                | 200         | 1200            |

- **Initial Strategy**: Applied SMOTE to balance all classes equally.
- **Refined Strategy**: Adjusted SMOTE weights for challenging categories like `entertainment` and `health and lifestyle`, reducing misclassifications.

---

## Model Development

### Stacking Classifier
An ensemble model was built using:
- **Base Model**: Random Forest (robust for high-dimensional TF-IDF features).
- **Meta Model**: Logistic Regression (efficient and interpretable).

### Cross-Validation
5-fold stratified cross-validation ensured consistent model performance.

### Hyperparameter Optimization
Bayesian optimization (via `scikit-optimize`) was used for efficient hyperparameter tuning, significantly reducing runtime.

---

## Results
The final model achieved improved accuracy and robustness, particularly in challenging and imbalanced classes, highlighting the importance of preprocessing, feature extraction, and careful balancing strategies.
<img width="425" alt="Ekran Resmi 2025-01-12 19 53 51" src="https://github.com/user-attachments/assets/78014f52-4612-46f8-8eab-fa7a5db890ec" />

The model achieved an average cross-validation accuracy of 0.9729, showcasing its ability to generalize well across different data splits. The confusion matrix highlights that most predictions align with their true labels, as seen by the dominant diagonal. Categories like "gaming" and "sports" were predicted perfectly, while minor misclassifications occurred in semantically similar classes, such as "entertainment" overlapping with "art" or "health and lifestyle.

<img width="662" alt="Ekran Resmi 2025-01-12 19 55 05" src="https://github.com/user-attachments/assets/d8400086-2439-4472-8b48-10f9ae0c777c" />

The ROC-AUC curve further validates the model’s robustness, with a weighted AUC of 0.9985. This indicates near-perfect separability across all categories.

# Like Count Regression

## Feature Extraction

The dataset included user and post-related features essential for predicting like counts. Textual features, such as the word count, emoji count, and hashtag count, were extracted from captions to quantify their impact on engagement. User-level features, including follower and following counts, average likes, and comments per user, were computed to capture overall engagement behavior. Boolean features (e.g., verified account, business account, private account) provided account-level characteristics, while highlight reel counts indicated user activity levels.

---

## Feature Engineering

A significant skewness was observed in the dataset for numerical features such as like counts, follower counts, and engagement metrics. To handle this, **log transformations** were applied to stabilize variance and normalize these features. Additionally, all numerical features were scaled using **Min-Max Normalization** to standardize the dataset for the regression model. New features were derived to provide a richer representation of the data:
- Ratios, such as `follower_following_ratio` and `emoji_per_word`, were calculated to explore interactions between variables.
- Interaction-based features, such as `business_follower` and `verified_business`, were engineered to combine boolean and numerical data.
- One-hot encoding was applied to categorical segments like follower count bins and caption length bins, further enhancing feature diversity.

---

## Model Training

A **Random Forest Regressor** was chosen for its robustness in handling high-dimensional data and feature interactions. The model was trained on log-transformed target values (`log1p` of like counts) to address the skewed distribution of the target variable. Key hyperparameters, such as the number of estimators and tree depth, were tuned to achieve optimal performance. The model initially predicted **float values**, which were then rounded to integers using the best rounding method based on validation performance. Among the evaluated rounding methods (`round`, `floor`, `ceil`, `int`), the **`round` method** was found to perform best, achieving a **Validation Log MSE of 0.5854**.

---

## Results

The model demonstrated excellent predictive performance, as illustrated in the scatter plot of predicted vs. true log-transformed like counts. Most predictions closely followed the perfect prediction line, indicating high accuracy across the dataset. The **log transformation** effectively mitigated skewness, leading to stable and reliable predictions. Minor deviations at higher engagement levels suggest some sensitivity to outliers, but overall, the model successfully generalized the patterns in the data.

<img width="463" alt="Ekran Resmi 2025-01-12 20 07 55" src="https://github.com/user-attachments/assets/c9095153-65c0-447d-a33c-96b430d32608" />

### Key Metrics
- **Validation Log MSE (Float Predictions)**: 0.5548
- **Validation Log MSE (Rounded Predictions)**: 0.5854
- **Validation R² Score**: 0.7314




