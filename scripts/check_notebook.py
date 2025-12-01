import nbformat as nbf

nb_path = 'sentiment_analysis.ipynb'
try:
    nb = nbf.read(nb_path, as_version=4)
    print("Notebook read successfully.")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and ('# ML-Based Sentiment Analysis' in cell['source'] or 'TfidfVectorizer' in cell['source']):
            print(f"--- Cell {i} Content ---")
            print(cell['source'])
            print("----------------------")
except Exception as e:
    print(f"Error reading notebook: {e}")
