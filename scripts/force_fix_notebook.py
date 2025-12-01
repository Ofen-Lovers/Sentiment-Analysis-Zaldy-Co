import nbformat as nbf

nb_path = 'sentiment_analysis.ipynb'
nb = nbf.read(nb_path, as_version=4)

# Robust ML Code
robust_ml_code = """# ML-Based Sentiment Analysis
# We use VADER and TextBlob labels as Ground Truth targets to compare how well ML models can learn them.

print("Handling NaN values...")
# Ensure no NaNs and all are strings
df['cleaned_comments'] = df['cleaned_comments'].fillna('')
df['cleaned_comments'] = df['cleaned_comments'].astype(str)

targets = ['vader_sentiment', 'textblob_sentiment']
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

X = df['cleaned_comments']
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X).toarray()

results_txt = ""

for target in targets:
    print(f"\\n{'='*40}\\nTarget: {target}\\n{'='*40}")
    results_txt += f"\\n{'='*40}\\nTarget: {target}\\n{'='*40}\\n"
    
    y = df[target]
    
    # Filter out NaN if any in target
    valid_idx = y.notna()
    X_subset = X_tfidf[valid_idx]
    y_subset = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

    for name, model in models.items():
        print(f"Training {name} on {target}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        
        results_txt += f"--- {name} ---\\n"
        results_txt += f"Accuracy: {acc:.4f}\\n"
        results_txt += f"Classification Report:\\n{report}\\n\\n"
        
        # Confusion Matrix
        unique_labels = sorted(y_subset.unique())
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f'Confusion Matrix - {name} ({target})')
        plt.ylabel(f'True Label ({target})')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}_{target}.png')
        plt.show()

with open('ml_results.txt', 'w') as f:
    f.write(results_txt)
"""

# Find and replace
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and ('# ML-Based Sentiment Analysis' in cell['source'] or 'TfidfVectorizer' in cell['source']):
        cell['source'] = robust_ml_code
        found = True
        break

if not found:
    print("ML cell not found. Appending...")
    nb['cells'].append(nbf.v4.new_code_cell(robust_ml_code))

nbf.write(nb, nb_path)
print("Notebook force-fixed successfully.")
