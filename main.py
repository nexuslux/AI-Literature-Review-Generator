import os
import PyPDF2
import openai
from dotenv import load_dotenv
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential
import unicodedata
import re

# Load environment variables and set up OpenAI client
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperSummary(BaseModel):
    title: str
    authors: List[str]
    year: int
    research_question: str
    theoretical_framework: str
    methodology: str
    main_arguments: List[str]
    findings: str
    significance: str
    limitations: str
    future_research: str

def clean_text(text: str) -> str:
    """Clean and normalize text to handle special characters."""
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Remove non-printable characters
    text = ''.join(char for char in text if ord(char) >= 32)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def analyze_pdf(text: str, filename: str) -> PaperSummary:
    """Analyze the content of a PDF and generate a structured summary."""
    prompt = f"""Analyze the following academic paper and provide a detailed summary in JSON format:

    Filename: {filename}
    Text: {text[:6000]}  # Limit text to 6000 characters

    Provide the summary in a structured JSON format with the following fields:
    - title: string
    - authors: array of strings
    - year: integer
    - research_question: string
    - theoretical_framework: string
    - methodology: string
    - main_arguments: array of strings
    - findings: string
    - significance: string
    - limitations: string
    - future_research: string"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides comprehensive academic summaries in JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    
    summary = PaperSummary.parse_raw(response.choices[0].message.content)
    return summary

def process_pdf(file_path: str) -> PaperSummary:
    """Process a single PDF file."""
    filename = os.path.basename(file_path)
    logger.info(f"Processing: {filename}")
    text = extract_text_from_pdf(file_path)
    return analyze_pdf(text, filename)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def synthesize_reviews(summaries: List[PaperSummary]) -> str:
    """Synthesize multiple paper summaries into a comprehensive literature review."""
    prompt = f"""Create a comprehensive literature review based on the following paper summaries. 
    Focus on synthesizing information, comparing and contrasting key arguments, methodologies, and significance of findings. 
    Highlight any contradictions, agreements, or trends between authors. 
    Discuss the evolution of ideas and methodologies in the field.
    Identify gaps in the current research and suggest future research directions.
    Keep the review under 2500 words.

    Summaries: {[summary.dict() for summary in summaries]}

    Structure the review as follows:
    1. Introduction
    2. Theoretical Frameworks
    3. Methodological Approaches
    4. Synthesis of Main Arguments and Findings
    5. Significance and Implications
    6. Gaps and Future Research Directions
    7. Conclusion"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates comprehensive, well-structured literature reviews."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000
    )
    return response.choices[0].message.content

def create_apa_citation(summary: PaperSummary) -> str:
    """Create an APA 7th edition style citation for a paper."""
    # Handle case where there are no authors
    if not summary.authors:
        return f"Unknown. ({summary.year}). {summary.title}."

    # Format authors: Last name, First initial. for all authors
    formatted_authors = []
    for author in summary.authors:
        parts = author.split()
        if len(parts) > 1:
            last_name = ' '.join(parts[-2:]) if parts[-2].lower() in ['van', 'von', 'de', 'du'] else parts[-1]
            initials = '. '.join(name[0].upper() + '.' for name in parts[:-1] if name.lower() not in ['van', 'von', 'de', 'du'])
            formatted_authors.append(f"{last_name}, {initials}")
        else:
            formatted_authors.append(author)
    
    # Join authors
    if len(formatted_authors) == 1:
        authors_string = formatted_authors[0]
    elif len(formatted_authors) == 2:
        authors_string = f"{formatted_authors[0]} & {formatted_authors[1]}"
    elif len(formatted_authors) > 2:
        authors_string = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
    else:
        authors_string = "Unknown"
    
    # Capitalize only the first word and proper nouns in the title
    title_words = summary.title.split()
    title = ' '.join([word.capitalize() if i == 0 or word.lower() not in ['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of'] else word.lower() for i, word in enumerate(title_words)])
    
    # Remove any trailing period from the title
    if title.endswith('.'):
        title = title[:-1]
    
    return f"{authors_string} ({summary.year}). {title}."

def create_paper_list(summaries: List[PaperSummary]) -> str:
    """Create a formatted list of reviewed papers with APA citations."""
    paper_list = "## List of Reviewed Papers\n\n"
    for summary in summaries:
        try:
            citation = create_apa_citation(summary)
            paper_list += f"- {citation}\n"
        except Exception as e:
            logger.error(f"Error creating citation for paper: {summary.title}. Error: {str(e)}")
            paper_list += f"- Error in citation: {summary.title}\n"
    return paper_list

def find_pdf_folder():
    """Find the 'PDF' folder in the same directory as the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, 'PDF')
    if os.path.isdir(pdf_folder):
        return pdf_folder
    else:
        raise FileNotFoundError("PDF folder not found in the script directory.")

def main():
    try:
        pdf_folder = find_pdf_folder()
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.error("No PDF files found in the PDF folder. Exiting.")
            return
        
        summaries = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_pdf, os.path.join(pdf_folder, pdf)) for pdf in pdf_files]
            for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Analyzing PDFs"):
                try:
                    summary = future.result()
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}")

        if not summaries:
            logger.error("No papers were successfully processed. Exiting.")
            return

        logger.info("Synthesizing literature review...")
        literature_review = synthesize_reviews(summaries)
        
        paper_list = create_paper_list(summaries)
        
        output_filename = f'literature_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(output_filename, 'w') as f:
            f.write(literature_review)
            f.write("\n\n")
            f.write(paper_list)
        
        logger.info(f"Literature review completed and saved as {output_filename}")
    
    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
