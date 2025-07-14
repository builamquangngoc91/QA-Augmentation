import os
import google.generativeai as genai
import json
import time
import re

# --- Step 1: Configure the Generative Model ---
api_key = ""
if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
    exit()
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')

# --- Step 2: Read the Source Text File ---
input_file_path = 'parsed_pathoma_output.md' 
print(f"Reading source text from '{input_file_path}'...")
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        full_text_from_file = f.read()
    print("Successfully read the source text file.")
except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found.")
    exit()

# --- Step 3: Define the Prompt Template (Không thay đổi) ---
rcc_classes_to_generate = [
    "Clear Cell Renal Cell Carcinoma (ccRCC)",
    "Papillary Renal Cell Carcinoma (pRCC)",
    "Chromophobe Renal Cell Carcinoma (chRCC)"
]
prompt_template = """
You are a pathology expert creating study materials.
First, carefully read the **Full Pathology Text** provided below.
Next, find all sentences and facts related to the specific disease: **{disease_name}**.
Based ONLY on the information you found about that disease within the provided text, generate exactly 100 high-quality question-and-answer pairs focused on key diagnostic and morphological features.
**Formatting Instructions:**
- The output MUST be a numbered list from 1 to 100.
- Each item must begin with "Q:" and the answer must start with "A:".
---
**Full Pathology Text:**
{source_text}
---
"""

# --- Step 4: Generate Q&A for Each Disease Class ---
final_knowledge_base = []
print("\nStarting Q&A generation with improved parsing logic...")

for disease in rcc_classes_to_generate:
    print(f"\n--- Generating for: {disease} ---")
    full_prompt = prompt_template.format(disease_name=disease, source_text=full_text_from_file)
    
    try:
        response = model.generate_content(full_prompt)
        
        # *** DÒNG CODE SỬA LỖI NẰM Ở ĐÂY ***
        # Regex này chấp nhận có hoặc không có số thứ tự ở đầu.
        qa_pairs = re.findall(r'(?:\d+\.\s*)?\**Q:\**\s*(.*?)\s*\**A:\**\s*(.*?)(?=\n\s*(?:\d+\.|\**Q:)|$)', response.text, re.DOTALL)

        if not qa_pairs:
            print(f"  - Warning: Model did not generate parsable Q&A for {disease}.")
            print("  - RAW MODEL RESPONSE WAS:\n", response.text)
            continue

        for q, a in qa_pairs:
            final_knowledge_base.append({
                "disease_class": disease,
                "question": q.strip(),
                "answer": a.strip()
            })
            
        print(f"  - Successfully generated and parsed {len(qa_pairs)} Q&A pairs.")
        time.sleep(2)

    except Exception as e:
        print(f"  - An error occurred while processing {disease}: {e}")

# --- Step 5: Save the Final Result ---
output_json_path = 'Final_RCC_QNA_From_File_Corrected.json'
print(f"\nSaving the generated knowledge base to '{output_json_path}'...")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_knowledge_base, f, indent=2, ensure_ascii=False)

print("\n--- Process Complete ---")
print(f"Final knowledge base with {len(final_knowledge_base)} Q&A pairs saved successfully.")