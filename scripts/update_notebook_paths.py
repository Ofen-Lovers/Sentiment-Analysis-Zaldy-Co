import nbformat as nbf
import re

nb_path = 'sentiment_analysis.ipynb'
nb = nbf.read(nb_path, as_version=4)

# Replacements
replacements = [
    ("outputs/cleaned_comments.csv", "outputs/data/cleaned_comments.csv"),
    ("outputs/lexicon_results.csv", "outputs/data/lexicon_results.csv"),
    ("outputs/ml_results.txt", "outputs/data/ml_results.txt"),
    ("outputs/sentiment_distribution.png", "outputs/images/sentiment_distribution.png"),
    ("outputs/confusion_matrix_", "outputs/images/confusion_matrix_")
]

print("Updating notebook paths...")
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        original_source = cell['source']
        new_source = original_source
        for old, new in replacements:
            # Simple string replacement might be risky if context matters, 
            # but for these specific filenames it should be safe.
            new_source = new_source.replace(f"'{old}", f"'{new}")
            new_source = new_source.replace(f'"{old}', f'"{new}')
            # Handle f-strings for confusion matrix
            if "confusion_matrix_" in old:
                 new_source = new_source.replace(f"f'{old}", f"f'{new}")
                 new_source = new_source.replace(f'f"{old}', f'f"{new}')
        
        if new_source != original_source:
            print(f"Updated cell:\n{original_source[:50]}...")
            cell['source'] = new_source

nbf.write(nb, nb_path)
print("Notebook paths updated successfully.")
