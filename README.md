# HEA_QA: High Entropy Alloy Question Answering System

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Details](#technical-details)
- [Setup Instructions](#setup-instructions)
- [Usage Instructions](#usage-instructions)
- [QA Pair Generation](#qa-pair-generation)
- [File Structure](#file-structure)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)
- [Model Card](#model-card-for-fine-tuned-phi-2)

## Overview

HEA_QA is an AI-powered question answering system designed to analyze research papers and provide insights related to High Entropy Alloys (HEAs). The system allows users to submit abstracts of scientific papers and ask questions about the materials, properties, and conditions described in those abstracts. By utilizing Retrieval-Augmented Generation (RAG), HEA_QA delivers accurate and contextually relevant answers based on the submitted content.

## Key Features

**Abstract Submission:**
- Support for both direct text input of single abstracts and batch upload via CSV files
- Automatic parsing of submitted abstracts to extract relevant information about alloys and their properties

**Question Answering:**
- Interactive interface for asking questions about submitted abstracts
- Utilizes fine-tuned language models to generate answers based on retrieved context
- Implements chunking and embedding techniques for efficient context retrieval

**Model Flexibility:**
- Toggle between a local fine-tuned model (Phi-2 with QLoRA) and API-based models (e.g., DeepSeek)
- Default configuration uses API model for higher accuracy with local model as fallback
- Optimized for domain-specific knowledge about High Entropy Alloys

**Document Management:**
- Persistent document store using Flask sessions and pickle
- Maintains processed abstracts and associated data between requests
- Includes functionality to clean old sessions and manage embeddings

**User Interface:**
- Clean, intuitive web interface built with Flask, HTML, CSS, and JavaScript
- Features include theme toggling (dark/light mode), cache cleaning, and chat history
- Responsive design for various screen sizes

**QA Pair Generation:**
- Utilizes Qwen2.5-1.5B-Instruct model to automatically generate question-answer pairs from abstracts
- Implements filtering using heuristics and model evaluation to ensure high-quality pairs
- Saves generated QA pairs for future reference and training

## Technical Details

**Backend Framework:**
- Flask (Python)

**Language Models:**
- Local: Fine-tuned Microsoft Phi-2 model with QLoRA (Low-Rank Adaptation)
- API: DeepSeek (accessed via OpenAI-compatible API)

**Vector Processing:**
- Sentence Embeddings: SentenceTransformer ('all-MiniLM-L6-v2')
- Vector Database: FAISS (Facebook AI Similarity Search)

**Core Dependencies:**
- transformers
- sentence-transformers
- flask
- flask-session
- faiss-cpu
- openai
- peft
- pandas

**Storage:**
- pickle for document persistence
- Flask sessions for user state management

**QA Generation:**
- Qwen2.5-1.5B-Instruct
- AutoModelForCausalLM
- AutoTokenizer

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ErAgOn-AmAnSiRoHi/HEA_QA
   cd HEA_QA
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set OpenAI API Key (if using API model):**
   - Obtain an API key from OpenAI or DeepSeek
   - Update the `api_key` in `app.py`:
     ```python
     client = OpenAI(
         base_url="https://api.deepseek.com",
         api_key="YOUR API-KEY HERE",
     )
     ```

4. **Download the phi2-lora-finetuned model (if not already present):**
   - Ensure the `models/phi2-lora-finetuned` directory exists and contains the necessary model files
   - If not, download it from [Hugging Face]([https://huggingface.co/models](https://huggingface.co/microsoft/phi-2)) 
   - Place the LoRA adapter in the `models/phi2-lora-finetuned` directory

5. **Run the Application:**
   ```bash
   python app.py
   ```

6. **Access the application:**
   - Open your web browser and navigate to `http://127.0.0.1:5000`

## Usage Instructions

1. **Submit an Abstract:**
   - Paste the abstract of a research paper into the "Submit Abstract" section
   - Alternatively, upload a CSV file containing multiple abstracts (one abstract per row, in a column labeled "Abstract")
   - Click the "Submit Abstract" button

2. **Ask Questions:**
   - Type your question in the chat input box
   - Click the send button
   - The system will retrieve relevant information from the submitted abstract and generate an answer

3. **View Extracted Data (optional):**
   - Click the "View Extracted Data" button to see a table of the parsed information from the abstract

4. **View QA Pairs (optional):**
   - Click the "See QA Pairs" button to see a table of the generated suggested question-answer pairs from the abstract

5. **Toggle Model:**
   - Use the "Toggle Model" button to switch between the local fine-tuned model and the API-based model

6. **Clean Cache:**
   - Use the "Clean Cache" button to clear the Hugging Face and Flask session caches.
   - [NOTE: Remember that this will remove hugging face .cache as well which include the downloaded model as well. So, use it only if the systems seems not to function properly.]

7. **Theme Toggle:**
   - Switch between dark and light mode using the theme toggle button

## QA Pair Generation [Optional]

The system includes a script for generating question-answer pairs from abstracts. [You don't need to run this separately, as running the flask app, runs this script in the bakground for QA Pairs generation]

1. **Navigate to the project directory:**
   ```bash
   cd HEA_QA
   ```

2. **Run the QA Pair Generation script:**
   ```bash
   python qa_pair_gen.py input.json --output_dir ./output/
   ```

   **Parameters:**
   - Replace `input.json` with the path to your JSON file containing alloy data
   - `--output_dir`: Directory to save output files
   - `--limit`: (Optional) Limit the number of contexts to process
   - `--rating_threshold`: (Optional) Set the minimum rating threshold for final filtering

The script uses the Qwen2.5-1.5B-Instruct model to generate QA pairs from abstracts, filters them using heuristics and model evaluation, and saves high-quality pairs to the specified output directory.

## File Structure

```
HEA_QA/
├── app.py             # Main Flask application file
├── models/
│   └── phi2-lora-finetuned/   # Directory containing the fine-tuned LoRA adapter
├── static/          # Contains static files (CSS, JavaScript)
│   ├── css/
│   │   └── style.css    # CSS stylesheet
│   └── js/
│       └── script.js    # JavaScript file for front-end functionality
├── templates/       # Contains HTML templates
│   └── index.html     # Main HTML template
├── requirements.txt # List of Python dependencies
├── qa_pair_gen.py   # QA Pair Generation script
├── test/            # test files
│   ├── test.csv       # CSV test file (can be uploaded through the Multiple Abstract Tab in the UI)
│   └── test.txt       # TXT test file (can be pasted in the text-area provided under the Single Abstract Tab in the UI)
└── README.md        # This file
```

## Contribution Guidelines

We welcome contributions to HEA_QA! Please follow these guidelines:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Write clear and concise commit messages
4. Submit a pull request with a detailed description of your changes

When contributing, please:
- Follow the existing code style and conventions
- Add appropriate documentation
- Include tests for new functionality
- Ensure all tests pass before submitting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


The fine-tuned model delivers improved performance for domain-specific queries related to High Entropy Alloys, their composition, properties, and applications in materials science.
The project is technically, domain-agnostic as some tweaking can be done and the system will be ready for use in other related domains.
