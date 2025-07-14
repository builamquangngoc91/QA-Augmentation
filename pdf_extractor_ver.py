# Import the necessary libraries
from llama_parse import LlamaParse
import os
import argparse

# --- Step 1: Setup the Parser ---
print("Setting up the LlamaParse client...")
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-1TyDel5J64RdZMOSSc3F4ciQCI1s4BbIHxAIHnMjK3wB0ERz"

# Instantiate the parser
parser = LlamaParse(
    result_type="markdown",
    verbose=True
)

# --- Step 2: Define the Prompt Templates ---

# Prompt for general text chunks
text_prompt_template = (
    "Rewrite the following pathology text into a more formal and comprehensive paragraph "
    "suitable for a pathology report. STATEMENT: {}"
)

# Prompt specifically for image captions
image_prompt_template = (
    "You are a pathology expert. The following is a caption for a pathology image. Based on your knowledge "
    "of this condition, expand this caption into a detailed description of the likely microscopic or "
    "gross findings in the image. CAPTION: {}"
)


# --- Step 3: Parse Command Line Arguments ---
parser_args = argparse.ArgumentParser(description="Extract and process PDF content")
parser_args.add_argument("pdf_file_path", help="Path to the PDF file to process")
parser_args.add_argument("out_file_path", help="Path for the output file")
args = parser_args.parse_args()

pdf_path = args.pdf_file_path
output_file_path = args.out_file_path
print(f"Attempting to parse '{pdf_path}'...")

try:
    documents = parser.load_data(pdf_path)
    print(f"Successfully parsed the PDF into {len(documents)} chunks.\n")

    # --- Step 4: Generate Prompts and Save to a Single File ---
    print(f"Generating prompts and saving to '{output_file_path}'...")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        # Loop through ALL parsed documents
        for i, doc in enumerate(documents):
            source_text = doc.text.strip()
            
            # Check if the chunk is an image caption
            if source_text.startswith("Fig."):
                full_prompt = image_prompt_template.format(source_text)
                f.write(f"--- IMAGE PROMPT (from Chunk {i}) ---\n")
                f.write(full_prompt)
                f.write("\n\n========================================\n\n")
            
            # Check if the chunk has other text content
            elif source_text:
                full_prompt = text_prompt_template.format(source_text)
                f.write(f"--- TEXT PROMPT (from Chunk {i}) ---\n")
                f.write(full_prompt)
                f.write("\n\n========================================\n\n")

    print(f"Process complete. All prompts have been saved to '{output_file_path}'.")

except Exception as e:
    print(f"An error occurred: {e}")