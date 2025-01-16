# Influencer Category Classification
# Data Preprocessing and Feature Extraction 

---

## Data Preprocessing

### Handling Missing Values

- **Features which have high percentage missing values(80%)** (e.g., `business_category_name`) were excluded.
- **Features which has personal information** (e.g., `biography`, `external_url`) replaced with placeholders like `"no biography"` and `"unknown"`.
- **Categorical features** were imputed using their mode.

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
   - Numbers were removed and text converted to lowercase.
   - Vectorized using **TF-IDF** with 1-gram and 2-grams (5000 Feature)

2. **Biographies**: 
   - Preprocessed similarly to posts.
   - Because biography vectors were more relevant, they were scaled by a factor of three, which raised the F1 score.

### Numerical and Categorical Features
- **Numerical features** (e.g., `follower_count`, `following_count`) were scaled using Min-Max Scaling.
- **Categorical features** (e.g., `is_verified`, `is_business_account`) were one-hot encoded.

### Combined Dataset
The model's capacity to generalize across various data types was enhanced by the combination of processed features from text, numerical, and categorical columns to produce a single dataset.

---

## Handling Class Imbalance

### Using SMOTE
The table shows that there was a notable class imbalance in the dataset.

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
An ensemble model was constructed:
- **Base Model**: Random Forest (robust for high-dimensional TF-IDF features).
- **Meta Model**: Logistic Regression.

The model performed consistently thanks to 5-fold stratified cross-validation.

### Hyperparameter Optimization
Using Bayesian optimization (through `scikit-optimize` library), runtime was greatly decreased through effective hyperparameter tuning. 

---

## Results
Preprocessing, feature extraction, and careful balancing techniques are crucial, as demonstrated by the final model's increased accuracy and robustness, especially in difficult and unbalanced classes.

<img width="425" alt="Ekran Resmi 2025-01-12 19 53 51" src="https://github.com/user-attachments/assets/78014f52-4612-46f8-8eab-fa7a5db890ec" />

With an average cross-validation accuracy of 0.9729, the model demonstrated strong generalization across various data splits. The dominant diagonal in the confusion matrix indicates that the majority of predictions match their actual labels. While small misclassifications happened in semantically related classes, such as "entertainment" overlapping with "art" or "health and lifestyle," categories like "gaming" and "sports" were accurately predicted.

<img width="662" alt="Ekran Resmi 2025-01-12 19 55 05" src="https://github.com/user-attachments/assets/d8400086-2439-4472-8b48-10f9ae0c777c" />

The model's robustness is further confirmed by the ROC-AUC curve, which has a weighted AUC of 0.9985. 

# Like Count Regression

## Feature Extraction

Features related to users and posts that are necessary for predicting like counts were included in the dataset. To measure their effect on engagement, textual elements from captions, including word count, emoji count, and hashtag count, were taken out. To capture overall engagement behavior, user-level features such as average likes, comments per user, and follower and following counts were calculated. While highlight reel counts showed user activity levels, boolean features (such as verified account, business account, and private account) offered account-level characteristics.

---


## Feature Engineering
<img width="411" alt="Ekran Resmi 2025-01-12 20 11 00" src="https://github.com/user-attachments/assets/a5899145-33b5-4d3c-911a-c0049dee9659" />
For numerical features like likes, followers, and engagement metrics, there was a noticeable skewness in the dataset. **Log transformations** were used to normalize these features and stabilize variance in order to deal with this. As you can see, our follower_count feature's distribution has stabilized. This improves the model's ability to forecast relationships (the target variable was also subjected to a log transformation). To further standardize the dataset for the regression model, **Min-Max Normalization** was used to scale all numerical features. To give the data a more comprehensive representation, new features were created:
- Ratios, such as `follower_following_ratio` and `emoji_per_word`, were calculated to explore interactions between variables.
- Interaction-based features, such as `business_follower` and `verified_business`, were engineered to combine boolean and numerical data.
- One-hot encoding was applied to categorical segments like follower count bins and caption length bins, further enhancing feature diversity.


---

## Model Training

The robustness of a **Random Forest Regressor** in managing feature interactions and high-dimensional data led to its selection. In order to address the target variable's skewed distribution, the model was trained using log-transformed target values (`log1p` of like counts). Important hyperparameters were adjusted to attain the best results, including the number of estimators and tree depth. Based on validation performance, the model's initial predictions of **float values** were rounded to integers using the optimal rounding technique. It was determined that the **`round` method** performed the best among the evaluated rounding techniques (`round`, `floor`, `ceil`, {int`), with a **Validation Log MSE of 0.5854**.

---

## Results

The scatter plot of predicted vs. true log-transformed like counts shows how well the model performed in terms of prediction. The perfect prediction line was closely followed by the majority of predictions, suggesting high accuracy throughout the dataset. By successfully reducing skewness, the **log transformation** produced predictions that were steady and trustworthy. The model generally did a good job of generalizing the patterns in the data, though slight variations at higher engagement levels indicate some sensitivity to outliers. Despite all this, we also see that the model has difficulty predicting posts with a true value of 0.

<img width="463" alt="Ekran Resmi 2025-01-12 20 07 55" src="https://github.com/user-attachments/assets/c9095153-65c0-447d-a33c-96b430d32608" />

- **Validation Log MSE (Float Predictions)**: 0.5548
- **Validation Log MSE (Rounded Predictions)**: 0.5854
- **Validation R² Score**: 0.7314




