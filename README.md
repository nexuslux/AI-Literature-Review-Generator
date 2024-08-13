# AI-Literature-Review-Generator
This project automates the process of generating comprehensive literature reviews from a collection of academic PDFs using OpenAI's GPT models.

# AI Literature Review Generator

This project automates the process of generating comprehensive literature reviews from a collection of academic PDFs using OpenAI's GPT models.

## Features

- Extracts text from PDF files
- Analyzes individual papers using AI
- Synthesizes multiple paper summaries into a cohesive literature review
- Generates APA-style citations for reviewed papers

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository:
```
https://github.com/nexuslux/AI-Literature-Review-Generator/
```

2. Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

3. Install required packages
```
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
LIke this: OPENAI_API_KEY=your_api_key_here

5. Place your PDF files in a folder in the 'PDF files' folder

6. Run code
```
python main.py
```

