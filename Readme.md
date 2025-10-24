# NER Pipeline with Entity Management

**Complete solution for Named Entity Recognition with global entity management**

## 🎯 Project Overview

This project implements a mini NER pipeline that extracts and manages named entities from technical texts using:
- **Classical NLP**: spaCy custom-trained model
- **LLM-based**: Google Gemini API with prompt engineering
- **Entity Management**: Global entity list with fuzzy matching and linking

### Core Challenge Solution
✅ Maintains a **growing, global list** of all extracted entities  
✅ Provides **existing entity list as context** to extraction  
✅ **Links** newly extracted entities to existing ones (fuzzy matching)  
✅ **Adds new** entities to the list  
✅ Handles **context window limits** for LLMs

---

## 📁 Project Structure

NER/
├── src/
│   ├── ner_classical.py          # Classical NER module (spaCy)
│   ├── ner_llms.py               # LLM NER module (Gemini)
│   ├── entity_manager.py         # Entity management system
│   └── ner_pipeline.py           # Main pipeline integrating all components
│   ├── build_dataset.py          # Extract text from pdf file 
│   ├── preprocess_texts.py       # Clean and extract command line
│   ├── analysis.py               # Analyze entity distribution
│   └── convert_format.py         # Convert text json file to spacy json
│
├── model_ner/                    # Trained spaCy model (generated after training)
│
├── results/                                # Output results
│   ├── test_file_classical_entities.json   # classical entities 
│   ├── test_file_classical_results.json    # classical results 
│   ├── test_file_llm_entities.json         # LLM entities
│   ├── test_file_llm_results.json          # LLM results
│   └── entity_database.json
│
├── data_train.json               # Training data (entity annotations)
├── data_dev.json                 # Development/test data
├── data_test.json                # Test data
├── test_file.json                # Sample input texts
├── coupling_relay_3RQ1_en-US.pdf # PDF file
├── test_file.json                # Sample input texts
├── instructions.json             # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # This file

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd ner-pipeline

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (if using pre-trained)
python -m spacy download en_core_web_sm
```

### 2. Setup API Key (for LLM)

```bash
# Get API key from: https://aistudio.google.com/apikey
export GEMINI