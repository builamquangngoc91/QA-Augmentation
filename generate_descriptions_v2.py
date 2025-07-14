import os
import google.generativeai as genai
import json
import re
import time

# --- Step 1: Configure the Generative Model ---
api_key = "AIzaSyAaI_G_UBCxSzn7hxhS8XS2zoCstwPpO0I"
if not api_key:
    print("Error: API_KEY not found.")
    exit()
genai.configure(api_key=api_key)

# **CORRECTION 1:** The model 'gemini-2.5-pro' does not exist.
# Using 'gemini-1.5-flash' which is valid and effective for this task.
model = genai.GenerativeModel('gemini-2.5-pro')


# --- Step 2: Load the Q&A Knowledge Base from JSON File ---
# **CORRECTION 2:** Using the filename you provided as the input.
input_json_path = 'Final_TCGA_Renal_QNA_Gemini_Only.json'
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

# This filter prompt works well and does not need changes.
filter_prompt_template = """
You are a pathology expert. I have a list of Q&A pairs.
Please analyze this list and return ONLY the Q&A pairs that are directly relevant to the following disease:
**{disease_name}**

Return the relevant pairs in the exact same format, starting with "Q:" and "A:". If none are relevant, return "No relevant Q&A found."

---
Full Q&A List:
{qa_list}
---
Relevant Q&A Pairs for {disease_name}:
"""

# **THIS IS THE NEW, IMPROVED PROMPT**
synthesis_prompt_template = """
You are a senior pathologist writing the "Microscopic Description" and "Summary" sections of a formal pathology report.
Your task is to convert the following list of Question-and-Answer pairs into a series of formal, descriptive statements about **{disease_name}**.

**CRITICAL INSTRUCTIONS:**
1.  Synthesize the information from the Q&A pairs into a list of 5-7 complete, well-phrased sentences.
2.  Each sentence must describe a distinct and important histopathologic feature.
3.  Ensure your descriptions cover a range of topics including **architecture, cytology (nuclei and cytoplasm), and vascular patterns** as detailed in the provided Q&A.
4.  Do not just copy the answers. Weave the facts into professional, descriptive prose.
5.  Output **only** the descriptive sentences, with each sentence on a new line. Do not use bullet points or numbers.

---
**Source Q&A Knowledge:**
{qa_context}
---
**Generated Formal Descriptions:**
"""

# --- Step 4: Generate Descriptions for Each Class ---
final_description_set = {}
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
        
        generated_descriptions = [line.strip() for line in synthesis_response.text.strip().split('\n') if line.strip()]
        
        final_description_set[disease] = generated_descriptions
        print(f"   - Successfully generated {len(generated_descriptions)} descriptions.")
        time.sleep(2) # Delay for API rate limits

    except Exception as e:
        print(f"   - An error occurred while processing {disease}: {e}")

# --- Step 5: Save the Final Result ---
output_json_path = 'Final_Generated_Descriptions.json'
print(f"\nSaving the synthesized descriptions to '{output_json_path}'...")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_description_set, f, indent=2, ensure_ascii=False)

print("\n--- Process Complete ---")
print("Final JSON file created successfully.")