# Sentiment Analysis: Rep. Zaldy Co Facebook Comments

This project analyzes public sentiment from **1,002 Facebook comments** regarding political discourse (specifically Rep. Zaldy Co). It compares lexicon-based sentiment classification with machine learning models.

## Key Features
- **Data Pipeline**: Includes translation (Tagalog/Cebuano/English), demojization, and NLTK-based text cleaning.
- **Sentiment Lexicons**: Utilizes **VADER** and **TextBlob** for automated labeling.
- **Machine Learning**: Trains and evaluates **Logistic Regression** and **Naive Bayes** models using TF-IDF features.
- **Comparative Analysis**: Evaluates the learnability of lexicon-based rules by ML models.

## Project Structure
- `sentiment_analysis.ipynb`: The main notebook containing the preprocessing, analysis, and modeling.
- `final_report.md`: A detailed report documenting the methodology, results, and conclusions.
- `data/`: Raw and processed comment datasets.
- `scripts/`: Utility scripts for data processing.
- `outputs/`: Visualizations and model performance metrics.

## Methodology Preview
The study uses VADER sentiment as a "ground truth" to evaluate how well ML models can learn social media sentiment patterns. Preprocessing is robust, handling multi-lingual data common in Filipino social media spaces.
