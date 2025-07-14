import os
import google.generativeai as genai
import json
import re

# --- Step 1: Configure the Generative Model ---
api_key = "AIzaSyAaI_G_UBCxSzn7hxhS8XS2zoCstwPpO0I"

if not api_key:
    print("Error: API_KEY not found.")
    exit()

genai.configure(api_key=api_key)
# Using a model with a larger context window and strong reasoning skills
model = genai.GenerativeModel('gemini-2.5-pro')


# --- Step 2: Load the Q&A Knowledge Base from JSON File ---
input_json_path = 'pathology_100_qna_morphology_focused.json' # Assuming this is your file with 100 Q&As
print(f"Loading Q&A knowledge base from '{input_json_path}'...")

try:
    with open(input_json_path, 'r', encoding='utf-8') as f:
        qa_knowledge_base = json.load(f)
    print(f"Successfully loaded {len(qa_knowledge_base)} Q&A pairs.")
except FileNotFoundError:
    print(f"Error: The input file '{input_json_path}' was not found.")
    exit()


# --- Step 3: Define Target Classes and Prompt Templates ---
rcc_classes = [
    "Clear Cell Renal Cell Carcinoma (ccRCC)",
    "Papillary Renal Cell Carcinoma (pRCC)",
    "Chromophobe Renal Cell Carcinoma (chRCC)"
]

# New prompt to ask the AI to act as a filter
filter_prompt_template = """
You are a pathology expert. I have a list of general renal pathology Q&A pairs.
Please analyze this list and return ONLY the Q&A pairs that are directly relevant to the following disease:
**{disease_name}**

Return the relevant pairs in the exact same format, starting with "Q:" and "A:". If none are relevant, return "No relevant Q&A found."

---
Full Q&A List:
{qa_list}
---
Relevant Q&A Pairs for {disease_name}:
"""

# Prompt to synthesize descriptions from the filtered Q&A pairs
synthesis_prompt_template = """
You are a pathologist writing a summary of key features.
Based ONLY on the provided Question-Answer pairs below, generate a list of 5 to 7 distinct, formal, one-sentence descriptions for: **{disease_name}**.
Each sentence should describe a key morphological, architectural, or diagnostic feature.
Do not use bullet points. Each description must be on a new line.

---
Relevant Q&A Pairs:
{qa_context}
---
Generated Descriptions:
"""

# --- Step 4: Generate Descriptions for Each Class ---
final_description_set = {}
# Convert the list of dicts into a simple string for the prompt
qa_list_string = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_knowledge_base])

print("\nStarting AI-powered filtering and description synthesis...")

for disease in rcc_classes:
    print(f" - Processing: {disease}...")
    
    # --- AI Filtering Step ---
    print("   - Asking AI to filter relevant Q&A...")
    filter_prompt = filter_prompt_template.format(disease_name=disease, qa_list=qa_list_string)
    
    try:
        response = model.generate_content(filter_prompt)
        relevant_qas_text = response.text

        if "no relevant q&a found" in relevant_qas_text.lower():
            print(f"   - AI reported no relevant Q&A for {disease}. Skipping.")
            continue
            
        print("   - Relevant Q&A found. Now synthesizing description...")
        
        # --- AI Synthesis Step ---
        synthesis_prompt = synthesis_prompt_template.format(disease_name=disease, qa_context=relevant_qas_text)
        synthesis_response = model.generate_content(synthesis_prompt)
        
        # Process the response into a list of strings
        generated_descriptions = [line.strip() for line in synthesis_response.text.strip().split('\n') if line.strip()]
        
        final_description_set[disease] = generated_descriptions
        print(f"   - Successfully generated {len(generated_descriptions)} descriptions.")

    except Exception as e:
        print(f"   - An error occurred while processing {disease}: {e}")

# --- Step 5: Save the Final Result ---
output_json_path = 'generated_rcc_descriptions.json'
print(f"\nSaving the synthesized descriptions to '{output_json_path}'...")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_description_set, f, indent=2)

print("\n--- Process Complete ---")
print("Final JSON file created successfully.")