import os
import google.generativeai as genai
import json
import time
import re

# --- Step 1: Configure the Generative Model ---
# Sử dụng API key của bạn.
api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyAaI_G_UBCxSzn7hxhS8XS2zoCstwPpO0I")

if not api_key:
    print("Error: GOOGLE_API_KEY not found.")
    exit()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro')


# --- Step 2: Read the Source Text from File ---
# **QUAN TRỌNG**: Đảm bảo file input là file văn bản gốc được parse từ PDF.
input_file_path = 'generated_prompts.txt'
print(f"Reading source text from '{input_file_path}'...")

try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        full_text_from_file = f.read()
    print("Successfully read the source text file.")

except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found.")
    exit()


# --- Step 3: Define the Target Disease Classes & Final Prompt Template ---
rcc_classes_to_generate = [
    "Clear Cell Renal Cell Carcinoma (ccRCC)",
    "Papillary Renal Cell Carcinoma (pRCC)",
    "Chromophobe Renal Cell Carcinoma (chRCC)"
]

# Prompt mới: Tập trung sâu vào các đặc điểm hình thái.
prompt_template = """
You are a leading pathologist creating a question bank for board exams. Your task is to analyze the provided **Full Pathology Text** and create 100 question-and-answer pairs for the disease specified below.

**Disease to Focus On:** {disease_name}

**CRITICAL INSTRUCTIONS - The Q&A pairs MUST focus on PHYSICAL DESCRIPTIVE FEATURES:**
1.  **Prioritize Morphology:** All questions must relate to the physical appearance of the cells and tissues.
2.  **Gross Morphology:** Include details about the macroscopic appearance, such as **color** (e.g., golden-yellow), **texture**, and **shape** of the tumor.
3.  **Microscopic Architecture:** Describe the arrangement of cells (e.g., nested, papillary, solid sheets, 'chicken-wire' vasculature).
4.  **Cellular Details (Cytology):** Focus on the appearance of individual cells, including:
    - **Cytoplasm:** Is it clear, eosinophilic (pink), granular, or pale?
    - **Nuclei:** Are they round, wrinkled ('raisinoid'), what is their grade, do they have halos?
    - **Cell Membranes:** Are they distinct ('plant-cell' like)?
5.  **Source Material:** All answers must be derived **EXCLUSIVELY** from the provided **Full Pathology Text**.

**Formatting:**
- The output must be a numbered list from 1 to 100.
- Each item must start with "Q:" and the answer must start with "A:".

---
**Full Pathology Text:**
{source_text}
---
"""

# --- Step 4: Generate Q&A for Each Disease Class using Source Text ---
final_knowledge_base = []
print("\nStarting Q&A generation with a strong focus on morphology...")

for disease in rcc_classes_to_generate:
    print(f"\n--- Processing: {disease} ---")
    
    full_prompt = prompt_template.format(disease_name=disease, source_text=full_text_from_file)
    
    try:
        response = model.generate_content(full_prompt)
        
        # Regex linh hoạt để xử lý các định dạng đầu ra khác nhau
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
        
        time.sleep(2) # Delay to respect API rate limits

    except Exception as e:
        print(f"  - An error occurred while processing {disease}: {e}")

# --- Step 5: Save the Final Result to a JSON File ---
output_json_path = 'Final_RCC_QNA_Morphology_Focused.json'
print(f"\nSaving the final knowledge base to '{output_json_path}'...")

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_knowledge_base, f, indent=2, ensure_ascii=False)

print("\n--- Process Complete ---")
print(f"Final knowledge base with {len(final_knowledge_base)} Q&A pairs saved successfully.")