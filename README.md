# AirBnB Listing Analysis

An end-to-end data science project that combines NLP, exploratory data analysis, and machine learning to understand what drives Airbnb listing prices across major U.S. cities. The project moves from raw messy data to a trained XGBoost regression model, with a detour through natural language processing to validate whether listing descriptions actually reflect what guests experience.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statements](#problem-statements)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technical Pipeline](#technical-pipeline)
- [Key Findings](#key-findings)
- [Models and Evaluation](#models-and-evaluation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Contributors](#contributors)

---

## Project Overview

As Airbnb continues to disrupt traditional hospitality, understanding the mechanics behind listing prices becomes increasingly valuable, for hosts trying to optimize revenue, for guests making booking decisions, and for platforms managing marketplace fairness. This project digs into a multi-city Airbnb dataset to answer three distinct but connected questions: do descriptions match reality, what makes a city's listings unique, and what features ultimately determine price.

The analysis is built in three stages. First, we clean and audit the data carefully, making deliberate choices about how to handle missingness rather than dropping rows or filling blindly. Second, we use NLP techniques including word clouds, CountVectorizer, and spaCy POS-pattern matching to extract meaning from free-text descriptions. Third, we build a predictive model using XGBoost with a Random Forest-based feature selection step to reduce dimensionality and improve generalization.

---

## Problem Statements

Three research questions drove this project:

**1. Does the property description match the actual property?**
Hosts write their own descriptions. We wanted to know if the language used in those descriptions aligns with the amenities and property types actually listed, or if there is a systematic gap between marketing language and ground truth.

**2. What makes Airbnb listings favorable across different cities?**
Guest preferences are not uniform. A solo traveler in NYC is not looking for the same thing as a family in Boston. We analyzed city-level patterns in descriptions and amenities to surface what guests in each market actually value.

**3. What drives the price of listings?**
This is the core predictive question. Using structured features like room type, bedrooms, bathrooms, review scores, and engineered amenity dummies, we trained a regression model to predict log-transformed prices.

---

## Dataset

- **Source**: Kaggle (Airbnb Listings Dataset)
- **Cities covered**: New York City, San Francisco, Boston, Los Angeles
- **Key columns**: `log_price`, `property_type`, `room_type`, `amenities`, `description`, `bedrooms`, `bathrooms`, `beds`, `number_of_reviews`, `review_scores_rating`, `city`, `host_identity_verified`
- **Target variable**: `log_price` (log-transformed listing price, used directly for regression)

---

## Project Structure

```
airbnb-listing-analysis/
│
├── AirBnB_Listing_Analysis.ipynb    # Main analysis notebook
├── AirBNB_Listing_Analysis.pptx     # Presentation summarizing findings
├── airbnb_analysis.py               # Standalone Python script (full pipeline)
├── Airbnb_Data.csv                  # Source dataset (not included in repo)
└── README.md
```

---

## Technical Pipeline

### 1. Data Understanding

The first step was profiling the raw dataset to understand what we were working with. We computed NaN counts and percentages for every column, which revealed that fields like `thumbnail_url`, `host_since`, `first_review`, `last_review`, and `zipcode` had either very high missingness or were not useful for price modeling. These were dropped early.

The `amenities` column arrived as a string-encoded set (e.g., `{WiFi,"Air conditioning",Kitchen}`) and required custom parsing before it could be analyzed or featurized.

### 2. Data Cleaning

We took a deliberate, column-specific approach to handling missing values rather than applying a single blanket strategy:

- **Numerical columns** (`bathrooms`, `bedrooms`, `beds`): Imputed with the column median. Median is more robust than mean when the distribution has outliers, which is common in listing data where one property might have 10 bedrooms while the typical one has 1 or 2.
- **`review_scores_rating`**: Split into two groups. For listings with zero reviews, we computed the mean rating among all zero-review listings separately and filled from that. For listings with reviews, we used the overall column mean. This avoids the bias that would come from mixing the two groups.
- **`host_identity_verified`**: Filled with `'f'` (unverified) as the conservative assumption for missing values.
- **Dropped columns**: `id`, `zipcode`, `thumbnail_url`, `last_review`, `first_review`, `host_response_rate`, `host_since`, `host_has_profile_pic`, `neighbourhood`, and the missingness-indicator columns created during imputation.

### 3. Exploratory Data Analysis

**Word Cloud of Descriptions**

We used `CountVectorizer` with English stop words removed to extract the most frequent meaningful terms across all listing descriptions. The resulting word cloud showed that the dominant themes are apartment-style accommodations (`apartment`, `private`, `bedroom`), proximity to food and services (`restaurant`, `walk`, `minutes`), and comfort features (`kitchen`, `space`, `comfortable`). This validated that descriptions are largely consistent with the types of properties listed and the amenities provided.

**Top Amenities Analysis**

We parsed the `amenities` column using a custom string parser, collected all amenity mentions across the dataset, and ranked them by frequency. The top amenities were:

- Internet / Wi-Fi (connectivity as a baseline expectation)
- Kitchen (cost-saving and flexibility for longer stays)
- Heating and Air Conditioning (comfort and climate control)
- Smoke detector and Carbon monoxide detector (safety essentials)
- Hair dryer and Shampoo (hotel-equivalent basics)

This shows that guests expect a baseline of comfort and safety, and hosts who do not provide these amenities are at a disadvantage in a competitive market.

**City-Level NLP with spaCy**

To understand what makes each city's listings distinct, we extracted phrase-level patterns from descriptions using spaCy's `Matcher` with POS tag sequences. Patterns like `[ADJ, NOUN]`, `[VERB, ADJ, NOUN]`, and `[NOUN, ADP, NOUN]` were used to surface characteristic phrases for each city.

We ran this on 100 listings per city (NYC, SF, Boston, LA) and compared the resulting frequency distributions visually using per-city word clouds with distinct colormaps. The patterns revealed meaningful differences: NYC descriptions emphasized neighborhoods and transit access, SF highlighted scenic views and walkability, Boston leaned into historic character, and LA focused on outdoor space and lifestyle.

### 4. Feature Engineering

The `amenities` column, once parsed into Python lists, was binary-encoded. Rather than creating dummies for every amenity (which produced hundreds of sparse columns and caused memory issues), we took the top 15 most frequent amenities and created one binary column per amenity. This reduced sparsity while retaining the most informative amenity signals.

One-hot encoding was applied to remaining categorical columns (`room_type`, `property_type`, `cancellation_policy`, `bed_type`, `host_identity_verified`) using `pd.get_dummies()` with `drop_first=True` to avoid multicollinearity.

### 5. Feature Selection

Before training XGBoost directly on the full feature set, we used a Random Forest Regressor as a feature selection step. Training a Random Forest with `n_estimators=50` and `max_depth=10` gave us feature importance scores across all columns. We then selected the top 10 most important features and evaluated model performance on this reduced set.

We also ran a sweep across feature counts (5, 10, 15 top features) and plotted MSE versus feature count to understand the performance-complexity tradeoff.

### 6. Predictive Modeling

We trained an XGBoost Regressor (`XGBRegressor`) with the following configuration:

- `n_estimators=70`
- `max_depth=5`
- `random_state=42`
- Target: `log_price` (log-transformed price, regressed directly)

The model was evaluated using:

- **Mean Squared Error (MSE)** on the held-out test set (20% split)
- **R-squared (R²)** as the primary accuracy metric, expressed as a percentage

A scatter plot of actual vs. predicted log prices with the ideal diagonal line was used to visualize prediction quality and identify any systematic bias at the high or low end of the price range.

---

## Key Findings

**Descriptions are consistent with reality.** The NLP analysis of listing descriptions showed that the most commonly used words and phrases align closely with the amenities and property types that hosts actually list. Guests can reasonably trust that what they read in a description reflects what they will find.

**City-specific preferences are real and distinct.** The spaCy-based phrase extraction revealed that each city has a characteristic set of terms that appear in its listings. This means a one-size-fits-all optimization strategy would not serve hosts in different markets equally. A Boston host should emphasize different attributes than an LA host.

**Amenities matter, but not all equally.** When featurized and passed through Random Forest feature importance, certain amenities (Wi-Fi, kitchen, AC/heating) consistently ranked as more predictive of price than others. Structural variables like `room_type`, `bedrooms`, and `accommodates` remained the dominant price drivers.

**XGBoost outperformed the baseline Random Forest** on the full feature set, with a lower MSE and higher R². The improvement was more pronounced when using the full feature set than the reduced one, suggesting that XGBoost's ability to handle higher-dimensional input with regularization was beneficial here.

---

## Models and Evaluation

| Model | Features Used | Relative MSE | Relative R² |
|---|---|---|---|
| Random Forest (baseline) | Top 5 features | Highest | Lowest |
| Random Forest (baseline) | Top 10 features | Moderate | Moderate |
| Random Forest (baseline) | Top 15 features | Lower | Higher |
| XGBoost | All features | Lowest | Highest |

The MSE-vs-feature-count plot showed diminishing returns beyond 10 features for Random Forest, while XGBoost benefited from the full feature space due to its regularization and boosting mechanism.

---

## How to Run

1. Clone the repository and navigate to the project folder.

2. Install dependencies (see below).

3. Place `Airbnb_Data.csv` in the root directory.

4. Run the notebook interactively:

```bash
jupyter notebook AirBnB_Listing_Analysis.ipynb
```

Or run the standalone script end-to-end:

```bash
python airbnb_analysis.py
```

The script will print model metrics to the console and save plots to the current directory.

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
wordcloud
spacy
```

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost wordcloud spacy
python -m spacy download en_core_web_sm
```

