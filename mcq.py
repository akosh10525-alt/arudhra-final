########################
# LOAD LIBRARIES
########################
import spacy
import warnings
import time
import logging

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from pdfminer.high_level import extract_text
from tqdm import tqdm

###################################################################################
# SUPPRESS WARNINGS THAT CAN CLUTTER THE OUTPUT, SUCH AS DEPRECATION WARNINGS, ETC.
###################################################################################
warnings.filterwarnings("ignore")

#####################################
# LOAD THE SPACY LANGUAGE MODEL.
#####################################
nlp = spacy.load("en_core_web_sm")

def nest_sentences(document, max_length=4096):
    """
    Break down a document into manageable chunks of sentences where each chunk is under a specified length.

    Parameters:
    - document (str): The input text document to be processed.
    - max_length (int): The maximum character length for each chunk.

    Returns:
    - list: A list where each element is a group of sentences that together are less than max_length characters.
    """
    nested = []  # List to hold all chunks of sentences
    sent = []    # Temporary list to hold sentences for a current chunk
    length = 0   # Counter to keep track of the character length of the current chunk
    doc = nlp(document)  # Process the document using Spacy to tokenize into sentences

    for sentence in doc.sents:
        length += len(sentence.text)
        if length < max_length:
            sent.append(sentence.text)
        else:
            nested.append(' '.join(sent))  # Join sentences in the chunk and add to the nested list
            sent = [sentence.text]  # Start a new chunk with the current sentence
            length = len(sentence.text)  # Reset the length counter to the length of the current sentence

    if sent:  # Don't forget to add the last chunk if it's not empty
        nested.append(' '.join(sent))

    return nested

def generate_summary(text, llm, max_length=4096):
    """
    Generate a summary for provided text using the specified large language model (LLM).

    Parameters:
    - text (str): Text to summarize.
    - llm (LLMChain): The large language model to use for generating summaries.
    - max_length (int): The maximum character length for each summary chunk.

    Returns:
    - str: A single string that is the concatenated summary of all processed chunks.
    """
    sentences = nest_sentences(text, max_length)
    summaries = []  # List to hold summaries of each chunk
    seen_questions = set()  # Set to track unique questions

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Generate diverse multiple-choice questions/answer are one sentence long based on the context here: {text}. "
                 "Ensure each question is unique and not repetitive. "
                 "Format:\nQuestion: Question?\n- A) Option A.\n- B) Option B.\n- C) Option C.\n- D) Option D.\nAnswer: Answer\n***\n"
                 "Generate five long answer question/answer, short answer question/answer , fill in the blanks with answer and true or false with answer "
    )

    for chunk in tqdm(sentences, desc="Generating summaries"):
        # Use the LLM to generate the summary based on the prompt.
        prompt = prompt_template.format(text=chunk)
        result = llm.invoke(prompt)
        result_lines = result.strip().split("\n")

        for line in result_lines:
            if line.startswith("Question:"):
                question = line.strip()
                if question not in seen_questions:
                    summaries.append(question)
                    seen_questions.add(question)
                    answer = next((l.strip() for l in result_lines if l.startswith("Answer:")), "")
                    summaries.append(answer)

        # Optionally print each generated summary.
        print(result.strip())

    # Join all summaries into a single string with spaces in between.
    return "\n".join(summaries)

def main_loop(delay):
    """
    Run the main loop, which generates summaries periodically, for 30 minutes.

    Parameters:
    - delay (int): The delay in seconds between each iteration of the loop.
    """
    end_time = time.time() + 30 * 60  # 30 minutes from now
    while time.time() < end_time:
        try:
            # Extract text from a PDF file.
            text = extract_text("C:/Users/Shruti/Desktop/Arudhra Project/veronica.pdf")

            # Generate and print the summary for the extracted text.
            summary = generate_summary(text, llm)
            print(summary)
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        # Pause for the specified delay before the next iteration.
        time.sleep(delay)

#####################################
# CONFIGURATION FOR THE LANGUAGE MODEL.
#####################################
llm = Ollama(model="llama3.2", temperature=0.9)

#######################################
# RUN THE MAIN LOOP FOR 30 MINUTES
#######################################
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the main loop for 30 minutes... Or whatever.")
    delay = int(input("Enter the delay time in seconds between each iteration: "))
    main_loop(delay)
    logging.info("Main loop completed.")