import spacy
import warnings
import logging
import os
from langchain.llms import Ollama
from pdf2image import convert_from_path
import easyocr
import gradio as gr
from reportlab.pdfgen import canvas
import pandas as pd  # Import pandas for Excel functionality
import random
import re

warnings.filterwarnings("ignore")

# Load Spacy modelc
nlp = spacy.load("en_core_web_sm")

# Load LLM
llm = Ollama(model="llama3", temperature=0.1)

# EasyOCR Reader
reader = easyocr.Reader(['en'],gpu=True)

import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# Specify the path to the Poppler bin directory
POPPLER_PATH = r"C:\Users\shali\Desktop\shaan_tyagi\poppler-24.08.0\Library\bin"  # Update this path to your Poppler bin directory

def pdf_to_images(file_path, start_page, end_page,dpi=150):
    """
    Convert PDF pages to images using pdf2image.
    """
    images = convert_from_path(file_path, first_page=start_page, last_page=end_page, fmt='jpeg',dpi=dpi, poppler_path=POPPLER_PATH)
    return images

def extract_text_from_images(images):
    """
    Extract text from images using EasyOCR.
    """
    extracted_text = ""
    for idx, img in enumerate(images):
        results = reader.readtext(img)
        for result in results:
            extracted_text += result[1] + " "
    return extracted_text

def generate_questions(extracted_text, question_counts, question_type):
    """
    Generate questions for each type with respective counts using the LLM
    and include specific keywords for the selected question type.
    """
    # Define a dictionary for question types and their keywords
    question_types_and_keywords = {

    "Comprehension": [
        "compare", "explain", "discuss", "describe", "differentiate", "analyze",
        "evaluate", "justify", "interpret", "elaborate", "define", "critique",
        "summarize", "illustrate", "assess","translate","contrast","classify",
        "discriminate","illustrate","detect error","rectify error",
        "identify relationship","extrapolate","interpolate","arrange in order"
    ],
    "Factual": [
        "state", "define", "list", "identify", "name", "mention", "outline",
        "recall", "label", "recognize", "specify", "classify", "enumerate",
        "highlight", "indicate"
    ],
    "Application": [
        "calculate", "predict", "analyze", "design", "solve", "construct",
        "formulate", "develop", "apply", "examine", "demonstrate", "compute",
        "modify", "synthesize", "implement","discover","devise a plan","set of operations",
        "select facts","select principle"
    ]
    }
    generated_output = ""
    for qtype, count in question_counts.items():
        if count > 0:  # Only generate if count is > 0

            # Fetch keywords based on the question type
            keywords_list = question_types_and_keywords.get(question_type, [])
            # Randomly select one keyword for question generation
            random_keyword = random.choice(keywords_list) if keywords_list else "explain"
            print("This is my question_type:", qtype)

            if question_type == "Application" :
                prompt = (
                    f"Generate {count} {qtype} questions that require **practical application and problem-solving**.\n"
                    f"When {qtype} is MCQ then generate the question with the four options and the answer\n"
                    f"Frame the questions using **real-world analysis and cause-effect relationships**.\n"
                    f"Use keywords: {random_keyword}.\n"
                    f"Example: 'If I increase the pulsewidth of a radar, how does it affect bandwidth and range?'\n\n"
                    f"Text to extract relevant concepts:\n{extracted_text}\n\n"
                    f"Now, provide **detailed answers** using logical reasoning. "
                    f"If exact information is not available, infer from the text without introducing external data."
                )
            elif question_type == "Comprehension":
                prompt = (
                    f"Generate {count} {qtype} questions that require **comprehension based questions based on the text**.\n"
                    f"Frame the questions using **how and why of any concept**.\n"
                    f"When {qtype} is MCQ then generate the question with the four options and the answer\n"
                    f"Use keywords: {random_keyword}, to generate the questions. \n"
                    f"Example: 'How do can we handle the blind speed in the moving target indicator radar by changing frequency.'\n\n"
                    f"Text to extract relevant concepts:\n{extracted_text}\n\n"
                    f"Now, provide **detailed answers** using logical reasoning. "
                    f"If exact information is not available, infer from the text without introducing external data."
                )
            else :
                prompt = (
                    f"Generate {count} {qtype} questions that require **factual based questions based on the text** include these keywords: {random_keyword}. "
                    f"Base these on the following text: {extracted_text}"
                    f"When {qtype} is MCQ then generate the question with the four options and the answer\n"
                    f"Example: 'What is blind speed in moving target indicator.'\n\n"
                    f"Now, generate answers to the following questions strictly based on the given text.\n"
                    f"You MUST generate an answer for EVERY question. If exact information is not found, summarize relevant content from the provided text.\n"
                    f"Ensure that you generate NO MORE and NO LESS than {count} questions.\n"
                    f"DO NOT generate answers outside this text. If the information is missing, say 'Answer not found in the provided text'.\n"
                )

            try:
                questions = llm.invoke(prompt)
                if not questions.strip():
                    raise ValueError("No answer generated by the model.")

                generated_output += f"--- {qtype} Questions ({count}) ---\n{questions}\n\n"

            except Exception as e:
                logging.error(f"Error generating answers: {e}")
                generated_output += f"--- {qtype} Questions & Answers ({count}) ---\n**Error: No valid answer generated.**\n\n"

    return generated_output

from reportlab.lib.pagesizes import letter

# def save_questions_to_pdf(output_text, file_name="generated_questions.pdf"):
#     from reportlab.lib.pagesizes import letter
#     from reportlab.pdfgen import canvas
#
#     c = canvas.Canvas(file_name, pagesize=letter)
#     width, height = letter
#     c.setFont("Helvetica", 12)
#
#     margin = 50
#     y = height - margin
#
#     # Extract metadata and questions
#     lines = output_text.split("\n")
#     metadata_lines = lines[:4]
#     question_lines = lines[5:]
#
#     # Center-align metadata (using slightly larger bold font)
#     c.setFont("Helvetica-Bold", 14)
#     for meta_line in metadata_lines:
#         text_width = c.stringWidth(meta_line, "Helvetica-Bold", 14)
#         c.drawString((width - text_width) / 2, y, meta_line)
#         y -= 25  # metadata line spacing
#
#     y -= 30  # Extra spacing after metadata
#
#     c.setFont("Helvetica", 12)
#
#     # Wrap and write each question line properly
#     max_width = width - 2 * margin
#
#     for line in question_lines:
#         if line.strip() == "" or "(0 words)" in line:
#             continue
#
#         # Check if page space is enough
#         if y < margin:
#             c.showPage()
#             c.setFont("Helvetica", 12)
#             y = height - margin
#
#         # Wrap text if too long
#         line_width = c.stringWidth(line, "Helvetica", 12)
#         if line_width > max_width:
#             words = line.split(" ")
#             current_line = ""
#             for word in words:
#                 test_line = f"{current_line} {word}".strip()
#                 if c.stringWidth(test_line, "Helvetica", 12) <= max_width:
#                     current_line = test_line
#                 else:
#                     c.drawString(margin, y, current_line)
#                     y -= 20
#                     current_line = word
#             if current_line:
#                 c.drawString(margin, y, current_line)
#                 y -= 20
#         else:
#             c.drawString(margin, y, line)
#             y -= 20
#
#     c.save()
#     return file_name

#  Previous code
# def save_questions_to_pdf(output_text, file_name="generated_questions.pdf"):
#     from reportlab.lib.pagesizes import letter
#     from reportlab.pdfgen import canvas
#     from reportlab.lib import utils
#
#     c = canvas.Canvas(file_name, pagesize=letter)
#     width, height = letter
#     c.setFont("Helvetica", 12)
#
#     margin = 50
#     y = height - margin
#
#     # Extract metadata and questions
#     lines = output_text.split("\n")
#     metadata_lines = lines[:4]
#     question_lines = lines[5:]
#
#     # Center-align metadata
#     for meta_line in metadata_lines:
#         text_width = c.stringWidth(meta_line, "Helvetica-Bold", 14)
#         c.drawString((width - text_width) / 2, y, meta_line)
#         y -= 30
#
#     c.setFont("Helvetica", 12)
#     y -= 30  # spacing
#
#     # Write questions (assuming they’re already clean now)
#     for line in question_lines:
#         if line.strip() == "" or "(0 words)" in line:
#             continue
#         if y < margin:
#             c.showPage()
#             c.setFont("Helvetica", 12)
#             y = height - margin
#
#         c.drawString(margin, y, line)
#         y -= 30
#
#     c.save()
#     return file_name

def save_questions_to_pdf(output_text, file_name="generated_questions.pdf"):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(file_name, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica", 12)

    margin = 50
    y = height - margin
    max_width = width - 2 * margin  # usable width

    # Extract metadata and questions
    lines = output_text.split("\n")
    metadata_lines = lines[:4]
    question_lines = lines[5:]

    # Center-align metadata
    for meta_line in metadata_lines:
        c.setFont("Helvetica-Bold", 14)
        text_width = c.stringWidth(meta_line, "Helvetica-Bold", 14)
        c.drawString((width - text_width) / 2, y, meta_line)
        y -= 30

    c.setFont("Helvetica", 12)
    y -= 20  # spacing

    # Helper function to wrap text
    def wrap_text(text, font_name, font_size, max_width):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if c.stringWidth(test_line, font_name, font_size) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    # Write questions with wrapping
    for line in question_lines:
        if line.strip() == "" or "(0 words)" in line:
            continue

        wrapped_lines = wrap_text(line, "Helvetica", 12, max_width)

        for wrapped_line in wrapped_lines:
            if y < margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = height - margin
            c.drawString(margin, y, wrapped_line)
            y -= 20  # line spacing

        y -= 10  # space between questions

    c.save()
    return file_name


# def save_questions_to_pdf(output_text, file_name="generated_questions.pdf"):
#     from reportlab.lib.pagesizes import letter
#     from reportlab.pdfgen import canvas
#
#     c = canvas.Canvas(file_name, pagesize=letter)
#     width, height = letter
#     c.setFont("Helvetica", 12)
#
#     margin = 50
#     y = height - margin
#
#     # Extract metadata and questions
#     lines = output_text.split("\n")
#     metadata_lines = lines[:4]
#     question_lines = lines[5:]
#
#     # Center-align metadata
#     for meta_line in metadata_lines:
#         text_width = c.stringWidth(meta_line, "Helvetica", 12)
#         c.drawString((width - text_width) / 2, y, meta_line)
#         y -= 30
#
#     c.setFont("Helvetica", 12)
#     y -= 30  # spacing
#
#     # Write questions (assuming they’re already clean now)
#     for line in question_lines:
#         if y < margin:  # Start a new page if space runs out
#             c.showPage()
#             c.setFont("Helvetica", 12)
#             y = height - margin
#
#         # Wrap text if it exceeds the width
#         line_width = c.stringWidth(line, "Helvetica", 12)
#         max_width = width - 2 * margin
#         if line_width > max_width:
#             words = line.split(" ")
#             current_line = ""
#             for word in words:
#                 test_line = f"{current_line} {word}".strip()
#                 if c.stringWidth(test_line, "Helvetica", 12) <= max_width:
#                     current_line = test_line
#                 else:
#                     c.drawString(margin, y, current_line)
#                     y -= 20
#                     current_line = word
#             if current_line:  # Draw the last part of the line
#                 c.drawString(margin, y, current_line)
#                 y -= 30
#         else:
#             c.drawString(margin, y, line)
#             y -= 20
#
#     c.save()
#     return file_name


def save_questions_to_excel(output_text, file_name="generated_questions.xlsx"):
    excel_path = os.path.abspath(file_name)
    lines = output_text.strip().split("\n")
    print("Extracted Lines:", lines)

    # **Extract metadata safely**
    try:
        course_name = lines[0].split(":")[1].strip()
        subject_code = lines[1].split(":")[1].strip()
        subject_name = lines[2].split(":")[1].strip()
        question_context_type = lines[3].split(":")[1].strip()
    except IndexError:
        print("Error: Metadata format is incorrect.")
        return None

    data = []
    i = 5
    current_qtype = "Unknown"

    while i < len(lines):
        line = lines[i].strip()

        # **Detect section headers**
        section_match = re.match(r"---\s*(.*?)\s*Questions", line)
        if section_match:
            current_qtype = section_match.group(1).strip()
            i += 1
            continue

        # **Extract question properly**
        if "Question" in line:
            question = line.split(":")[1].strip() if ":" in line else line.strip()
            i += 1

            # **Extract options (A, B, C, D)**
            options = []
            while i < len(lines) and "Answer:" not in lines[i]:
                if re.match(r"^[A-D]\)", lines[i]):  # Checking options correctly
                    options.append(lines[i].strip())
                i += 1

            # **Extract answer properly**
            answer = ""
            if i < len(lines) and "Answer:" in lines[i]:
                answer = lines[i].split(":")[1].strip() if ":" in lines[i] else lines[i].strip()

            # **Append data into list**
            data.append([
                course_name,
                subject_code,
                subject_name,
                question_context_type,
                current_qtype,
                question,
                ", ".join(options),  # Storing all options in one cell
                answer
            ])
        i += 1

    # **Define Columns**
    columns = [
        "Course Name",
        "Subject Code",
        "Subject Name",
        "Type of Question",
        "Question Type",
        "Question",
        "Options",
        "Answer"
    ]

    # **Check extracted data**
    print("Final Data to be Saved:", data)

    # **Save to Excel**
    if data:
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(excel_path, index=False)
        print(f"Excel file saved successfully at {excel_path}")
    else:
        print("No valid questions found in the data!")

    return excel_path


def process_pdf(file, start_page, end_page, question_counts, question_type):
    """
    Complete pipeline: Convert PDF to text and generate questions.
    """
    print("QUESTION COUNTS ",question_counts)
    if not file:
        return "No file uploaded. Please upload a PDF.", ""

        # **Check if all question counts are zero**
    if all(count == 0 for count in question_counts.values()):
        return "Please select the count for at least one question type.", ""

    try:
        # Save uploaded file
        file_path = file.name
        
        # Step 1: Convert PDF pages to images
        images = pdf_to_images(file_path, start_page, end_page)
        
        # Step 2: Extract text from images
        extracted_text = extract_text_from_images(images)

        if not extracted_text.strip():
            return "No text found. Ensure pages contain readable text.", ""

        # Step 3: Generate Questions
        generated_questions = generate_questions(extracted_text, question_counts, question_type)
        
        return "Text extraction and question generation successful!", generated_questions

    except Exception as e:
        logging.error(f"Error: {e}")
        return f"An error occurred: {e}", ""

custom_css = """
    .gradio-container {
        background-color: #e3f2fd; /* Light Blue Background */
        font-family: 'Inter', sans-serif;
    }
    .title {
        color: #002147;
        text-align: center;
        font-size: 32px; /* Increased Font Size */
        font-weight: bold;
        margin-bottom: 15px;
    }
    .subtitle {
        color: #333333;
        text-align: center;
        font-size: 20px; /* Increased Font Size */
        margin-bottom: 20px;
    }
    .card-section {
        background-color: #ffffff;
        box-shadow: 0px 6px 10px rgba(0, 0, 0, 0.2); /* Darker Border */
        padding: 22px;
        border-radius: 8px;
        border: 2px solid #004085; /* Darker Card Borders */
        margin-bottom: 15px;
    }
    .output-box {
        background-color: #e0e7f7; /* Darker Inner Box Background */
        border: 2px solid #002147; /* Darker Border */
        padding: 20px;
        border-radius: 5px;
        min-height: 320px; /* Slightly Larger */
        font-size: 18px; /* Larger Font */
        font-weight: bold; /* Darker, Bolder Text */
        color: #002147; /* Darker Text Color */
    }
    .primary-button {
        background-color: #002147 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 14px 22px; /* Slightly Larger Padding */
        border-radius: 5px;
        transition: all 0.3s ease-in-out;
    }
    .primary-button:hover {
        background-color: #001634 !important;
    }
    .secondary-button {
        background-color: #004085 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 14px 22px;
        border-radius: 5px;
        transition: all 0.3s ease-in-out;
    }
    .secondary-button:hover {
        background-color: #003366 !important;
    }
    .hover-dropdown select:hover {
        background-color: #f0f0f0;
        cursor: pointer;
    }
"""

with gr.Blocks(css=custom_css) as demo:

    # Header Section
    gr.Image("logo.jpg", elem_id="logo", show_label=False, interactive=False)
    gr.Markdown("<h1 class='title'>AKOSH1.1 AFTC Question Repository</h1>")
    gr.Markdown("<p class='subtitle'>Upload a PDF, select page range, and generate AI-based questions.</p>")

    with gr.Row():
        with gr.Column(scale=2, min_width=300):
            with gr.Group(elem_classes="card-section"):
                gr.Markdown("### Upload & Configure")

                file_input = gr.File(label="Upload PDF", type="filepath")

                # Dropdown for Course Name
                course_name_dropdown = gr.Dropdown(
                    label="Course Name",
                    choices=["Ab Initio", "PKC"],
                    value="Ab Initio",
                    interactive=True,
                )

                # Dropdown for Subject Code
                subject_code_dropdown = gr.Dropdown(
                    label="Subject Code",
                    choices=["1.1.1", "1.1.2", "1.1.3", "1.1.4", "1.1.5", "1.1.6", "1.1.7", "1.1.8"],
                    value="1.1.1",
                    interactive=True,
                )

                # Dropdown for Subject Name
                subject_name_dropdown = gr.Dropdown(
                    label="Subject Name",
                    choices=[
                        "Radar and Microwave Theory", "Communication Engineering", "Advanced Computer Network",
                        "Network Centric Warfare", "Artificial Intelligence", "Navigation and Airfield Aids",
                        "Maintenance ERP", "Aerospace Safety"
                    ],
                    value="Radar and Microwave Theory",
                    interactive=True,
                )

                # Dropdown for question type selection
                question_type_dropdown = gr.Dropdown(
                    label="Question Type",
                    choices=["Comprehension", "Factual", "Application"],
                    value="Comprehensive",
                    interactive=True,
                    elem_classes="hover-dropdown"
                )

                start_page_input = gr.Number(label="Start Page", value=1, precision=0)
                end_page_input = gr.Number(label="End Page", value=1, precision=0)

            with gr.Group(elem_classes="card-section"):
                gr.Markdown("### Select Question Types & Counts")
                question_types = ["MCQs", "Fill in the Blanks", "True/False", "Long Answer", "Short Answer", "Very Short Answer"]
                question_count_inputs = {qtype: gr.Number(label=f"{qtype} Count", value=0, precision=0) for qtype in question_types}

        with gr.Column(scale=3):
            with gr.Group(elem_classes="card-section output-box"):
                output_status = gr.Textbox(label="Status", interactive=False)
                output_questions = gr.Textbox(label="Generated Questions", lines=15)

            submit_button = gr.Button("Generate Questions", elem_classes="primary-button")

            with gr.Row():
                download_button = gr.Button("Download PDF", elem_classes="secondary-button")
                download_file = gr.File(label="PDF File", interactive=False)

            with gr.Row():
                download_excel_button = gr.Button("Download Excel", elem_classes="secondary-button")
                download_excel_file = gr.File(label="Excel File", interactive=False)

    generated_text = gr.State()

    # **Functions to Save PDF & Excel**
    def save_to_pdf_for_download(questions_text):
        pdf_file_path = "generated_questions.pdf"
        save_questions_to_pdf(questions_text, pdf_file_path)
        return pdf_file_path

    def save_to_excel_for_download(questions_text):
        excel_file_path = "generated_questions.xlsx"
        save_questions_to_excel(questions_text, excel_file_path)
        return excel_file_path

    def handle_generation(file, start_page, end_page, course_name, subject_code, subject_name, question_type, *args):
        import re
        question_counts = {qtype: args[idx] for idx, qtype in enumerate(question_types)}

        status, questions = process_pdf(file, start_page, end_page, question_counts, question_type)

        # Prepare centered metadata (to be center-aligned later in PDF)
        metadata_lines = [
            f"Course Name: {course_name}",
            f"Subject Code: {subject_code}",
            f"Subject Name: {subject_name}",
            f"Question Type: {question_type}"
        ]

        formatted_output = "\n".join(metadata_lines) + "\n"

        # Add optional separator
        formatted_output += "-" * 40 + "\n"

        for line in questions.strip().split("\n"):
            cleaned_line = re.sub(r"\s*\(\d+\s+words\)$", "", line).strip()
            if cleaned_line:
                formatted_output += cleaned_line + "\n"

        return status, formatted_output, formatted_output


    # **Button Click Events**
    submit_button.click(
        fn=handle_generation,
        inputs=[file_input, start_page_input, end_page_input, course_name_dropdown, subject_code_dropdown, subject_name_dropdown, question_type_dropdown] + list(question_count_inputs.values()),
        outputs=[output_status, output_questions, generated_text]
    )

    download_button.click(
        fn=save_to_pdf_for_download,
        inputs=[generated_text],
        outputs=[download_file]
    )

    download_excel_button.click(
        fn=save_to_excel_for_download,
        inputs=[generated_text],
        outputs=[download_excel_file]
    )

demo.launch()
