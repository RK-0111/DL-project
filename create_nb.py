import nbformat as nbf

nb = nbf.v4.new_notebook()

with open('train_models.py', 'r') as f:
    code = f.read()

nb['cells'] = [
    nbf.v4.new_markdown_cell('# Train and Evaluate All Models\nRun this cell to build, train, and plot results for all 4 models (MLP, ResNet-1D, CNN-LSTM, Transformer).'),
    nbf.v4.new_code_cell(code)
]

nbf.write(nb, 'Train_And_Evaluate_All.ipynb')
print("Notebook created successfully.")
