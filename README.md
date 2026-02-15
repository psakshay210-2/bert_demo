# BERT Fill-in-the-Blank Demo

This is a simple project to demonstrate the capabilities of **BERT (Bidirectional Encoder Representations from Transformers)**.

BERT is designed to understand the context of words in a sentence by looking at the words that come before and after it simultaneously. This demo showcases its **Masked Language Modeling (MLM)** ability, where it predicts a missing word based on the surrounding context.

## How to Run

### Prerequisities
You need Python installed on your system.

### 1. Install Dependencies
Open your terminal or command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Script
Execute the python script:
```bash
python bert_fill_mask.py
```

### 3. Usage
When prompted, type a sentence with the special token `[MASK]`. BERT will try to guess what word belongs there.

**Examples:**
- `The doctor ran to the [MASK].` -> BERT might predict: `hospital`, `patient`, `room`.
- `The [MASK] ran to the hospital.` -> BERT might predict: `doctor`, `nurse`, `patient`.
- `Paris is the [MASK] of France.` -> BERT might predict: `capital`.

## Understanding the Output
The script outputs a list of predicted words along with a **Confidence Score**.
- **Word**: The token BERT thinks fits best.
- **Confidence**: A percentage indicating how sure BERT is about that specific word.

## Offline Usage
**Does it need internet?**
## Deep Dive Notebook
For those who want to see the **internal workings** of BERT (Attention maps, Tokenization, Probability scores), we have provided a Jupyter Notebook.

### How to Run
1.  Install valid Jupyter support: `pip install jupyter matplotlib seaborn`
2.  Launch the notebook:
    ```bash
    jupyter notebook bert_inner_workings.ipynb
    ```
3.  Calculations and visualizations will run step-by-step.
