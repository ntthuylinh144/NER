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
```
NER/
├── src/
│   ├── ner_pipeline.py           # Main pipeline integrating all components
│   ├── ner_llm_model.py          # LLM NER module (Gemini)
│   ├── ner_classical_model.py    # Classical NER module (spaCy)
│   ├── manage_entities.py        # Entity management system
│   ├── build_dataset.py          # Extract text from pdf file
│   ├── text_preprocessing.py     # Clean and extract command line
│   ├── llm_annotate.py           # Use LLM to annotate
│   ├── convert_format.py         # Convert text json file to spacy json
│   ├── split_dataset.py          # Split dataset into train, dev, test
│   └── analysis.py               # Analyze entity distribution
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
├── data_dev.json                 # Development data
├── data_test.json                # Test data
├── test_file.json                # Sample input texts
├── coupling_relay_3RQ1_en-US.pdf # PDF file
├── test_file.json                # Sample input texts
├── instructions.json             # Dataset format json
├── spacy_ready.json              # Dataset format spacy
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```
---

## 🚀 Quick Start

### 1. Installation

```bash
# 1️⃣ Clone project from GitHub
git clone https://github.com/ntthuylinh144/NER.git
cd NER

# 2️⃣ Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 3️⃣ Install all dependencies
pip install -r requirements.txt

# 4️⃣ Download spaCy English model
python -m spacy download en_core_web_sm

# 5️⃣ Set your Gemini API Key
#    👉 Get your key from: https://aistudio.google.com/apikey
setx GEMINI_API_KEY "YOUR_GEMINI_API_KEY"

# ⚠️ Important: After running `setx`, close and reopen your terminal
# so the environment variable takes effect.

# 6️⃣ Run the complete NER pipeline
python src\ner_pipeline.py

==============================================
⚙️ ABOUT THE DATA
==============================================
The dataset used in this project was partially annotated
using an LLM (Gemini API) and then manually reviewed
and corrected for quality assurance.

✅ Therefore, you DO NOT need to rerun the preprocessing scripts:
    - build_dataset.py
    - text_preprocessing.py
    - llm_annotate.py
    - convert_data_format.py
    - split_dataset.py

 These scripts were only required during the initial dataset
 construction phase. The preprocessed and labeled dataare
 already included and ready for training/evaluation.

==============================================
🧩 Entity Definitions
============================================== 
| Entity Type | Description                                                                  | Example                                                            |
| ----------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| COMPONENT   | Physical parts or hardware elements involved in assembly.                    | servo motor, gripper, gear, screw, bracket, PCB, base plate, cable |
| TOOL        | Tools or instruments used to perform assembly actions.                       | screwdriver, wrench, soldering iron, multimeter                    |
| ACTION      | Technical operations or procedures performed on components.                  | attach, tighten, connect, calibrate, install, power on             |
| PARAMETER   | Technical parameters, measurements, or numerical values used in the process. | 5V, 10mm, 30 °C, torque = 5 Nm                                     |
| LOCATION    | Physical or relative positions mentioned in the assembly steps.              | left side, base, top, rear panel, slot A                           |
| MATERIAL    | Materials or substances used in the assembly or fabrication process.         | aluminum, plastic, copper wire, adhesive, silicone                 |
