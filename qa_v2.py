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

# Prompt mới: Kết hợp việc tìm kiếm thông tin trong source_text và tạo Q&A.
prompt_template = """
You are a pathology expert tasked with creating study materials.
First, carefully read the **Full Pathology Text** provided below.
Next, find all sentences and facts related to the specific disease: **{disease_name}**.

Based ONLY on the information you found about that disease within the provided text, generate exactly 10 distinct, high-quality question-and-answer pairs.

**CRITICAL INSTRUCTIONS:**
1.  Your answers must be derived ONLY from the **Full Pathology Text**. Do not use any outside knowledge.
2.  The Q&A pairs must focus on key features like morphology (gross and microscopic), color, size, cytology, architecture, and differential diagnosis as mentioned in the text.

Format the entire output as a numbered list. Each item must start with "Q:" and the answer must start with "A:".

---
**Full Pathology Text:**
{source_text}
---
"""

# --- Step 4: Generate Q&A for Each Disease Class using Source Text ---
final_knowledge_base = []
print("\nStarting Q&A generation for each class based on file content...")

for disease in rcc_classes_to_generate:
    print(f"\n--- Processing: {disease} ---")
    
    # Format the prompt with the current disease name and the full text from the file
    full_prompt = prompt_template.format(disease_name=disease, source_text=full_text_from_file)
    
    try:
        # This is a large prompt, so the model may take a moment
        response = model.generate_content(full_prompt)
        
        # Parse the Q&A pairs from the response
        qa_pairs = re.findall(r'\d+\.\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\n\d+\.|$)', response.text, re.DOTALL)
        
        if not qa_pairs:
            print(f"  - Warning: Model did not generate parsable Q&A for {disease}. The text might lack specific info.")
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
output_json_path = 'Final_RCC_QNA_from_File.json'
print(f"\nSaving the final knowledge base to '{output_json_path}'...")

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_knowledge_base, f, indent=2, ensure_ascii=False)

print("\n--- Process Complete ---")
print(f"Final knowledge base with {len(final_knowledge_base)} Q&A pairs saved successfully.")