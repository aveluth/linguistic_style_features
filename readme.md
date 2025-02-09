# Linguistic Feature Extraction and LIWC Analysis

A Python repository for extracting various linguistic and readability features from text data stored in CSV files. The code processes text data using natural language processing tools to compute metrics such as word count, sentence count, syllable count, lexical diversity, readability indices, sentiment probabilities, named entity counts, abstractness scores, and more. Additionally, if LIWC analysis data is available, the repository can compute combined stylometric features.

---

## Features

- **Linguistic Feature Extraction:**  
  Extracts word counts, sentence counts, syllable counts, average syllables per word, and lexical diversity.

- **Part-of-Speech (POS) Analysis:**  
  Computes normalized counts for various POS tags (e.g., NOUN, VERB, ADJ).

- **Readability Metrics:**  
  Calculates multiple readability indices (Flesch-Kincaid Grade, SMOG Index, Coleman-Liau Index, etc.) using [textstat](https://github.com/shivam5992/textstat).

- **Sentiment Analysis:**  
  Performs sentiment analysis for English (using [VADER](https://github.com/cjhutto/vaderSentiment)) and German (using [germansentiment](https://github.com/cjbarron/german-sentiment)) texts.

- **Named Entity Extraction:**  
  Identifies and counts named entities across categories such as people, temporal, spatial, and quantity-related entities.

- **Abstractness Features:**  
  Computes abstractness scores from provided abstractness datasets.

- **LIWC and Stylometric Features:**  
  If LIWC analysis data is available, the code computes combined stylometric features (e.g., self-reference, perceptual details).

---

## Requirements

- **Python Version:** Python 3.8+

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/linguistic_style_features.git
   cd linguistic_style_features
   ```

2. **(Optional) Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Create a `requirements.txt` file with the required packages or install manually:

   ```bash
   pip install pandas spacy textstat germansentiment vaderSentiment pyphen nltk tqdm
   ```

4. **Download SpaCy Models and NLTK Data:**

   ```bash
   python -m spacy download en_core_web_md
   python -m spacy download de_core_news_md
   python -c "import nltk; nltk.download('stopwords')"
   ```

---

## Usage

The main entry point for processing text data is `main.py`. It reads a CSV file, processes a specified text column, computes various features, and writes the output to a CSV file.

### Command-Line Arguments

- `--input`: Path to the input CSV file.
- `--liwc_analysis_included`: Include this flag if the input CSV contains LIWC analysis columns.
- `--text_column`: The name of the column in the CSV that contains text data.
- `--language`: The language of the text (`en` for English, `de` for German).
- `--output`: Path to the output CSV file.

### Example

```bash
python main.py --input data/input.csv --text_column content --language en --output data/output.csv
```

If your input CSV already contains LIWC analysis data, include the flag:

```bash
python main.py --input data/input.csv --liwc_analysis_included --text_column content --language de --output data/output.csv
```

---

## File Structure

```
text-feature-extraction/
├── data/
│   └── abstractness/
│       ├── 13428_2013_403_MOESM1_ESM.xlsx      # English abstractness data
│       └── ratings_lrec16_koeper_ssiw.txt       # German abstractness data
├── main.py                # Main script to process the CSV file.
├── feature_extraction.py  # Module for feature extraction and NLP processing.
├── utils.py               # Utility functions for CSV read/write (if applicable).
├── README.md              # This file.
└── requirements.txt       # List of dependencies (if provided).
```

---


