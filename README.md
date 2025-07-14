# Pathology PDF to TCGA-Style Description Generator

This system extracts text from pathology PDFs and generates TCGA-style descriptions using VQA/QA-based augmentation as a reasoning step.

## Overview

The pipeline consists of three main components:

1. **PDF Text Extraction** (`pdf_extractor.py`): Extracts relevant text from pathology PDFs
2. **VQA Augmentation** (`vqa_augmentation.py`): Generates question-answer pairs as reasoning steps
3. **Description Generation** (`description_generator.py`): Converts QA pairs into TCGA-style descriptions

## Architecture

```
PDF Input → Text Extraction → VQA Reasoning → TCGA Descriptions
```

### Knowledge-Based Module - Text Prior Branch
- Selects relevant chapters from PDF (e.g., "Kidney and Urinary Tract Pathology")
- Extracts text using LlamaParse for superior document understanding
- Provides knowledge-based content as teacher knowledge

### VQA Augmentation Module
- **Objective**: Improve quality and richness of pathology descriptions
- **Process**:
  1. Generate pathology-style questions from extracted text
  2. Create answers using context and reasoning
  3. Use QA pairs to write richer pathology captions
- **Benefits**:
  - Uncovers missing details
  - Adds structure and standard phrasing
  - Makes descriptions more interpretable

## Installation

```bash
pip install -r requirements.txt
```

**Important**: You need a LlamaCloud API key to use this system. Get one from [LlamaCloud](https://cloud.llamaindex.ai/) and set it as an environment variable:

```bash
export LLAMA_CLOUD_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
python main_pipeline.py "Pathoma 2021.pdf"
```

### Advanced Usage

```bash
python main_pipeline.py "Pathoma 2021.pdf" --output-dir results --config config.json
```

### With API Key Argument

```bash
python main_pipeline.py "Pathoma 2021.pdf" --llama-api-key your-api-key-here
```

### Configuration

Create a `config.json` file:

```json
{
  "output_dir": "output",
  "entities": [
    "Clear Cell RCC",
    "Papillary RCC",
    "Chromophobe RCC"
  ],
  "keywords": [
    "renal cell carcinoma", "clear cell", "papillary", "chromophobe",
    "kidney", "urinary tract", "RCC", "ccRCC", "pRCC", "chRCC",
    "cytoplasm", "nuclei", "architecture", "morphology"
  ],
  "save_intermediate": true,
  "use_llm_augmentation": false
}
```

## Output Format

The system generates descriptions in TCGA format similar to `TCGA_RCC_desc_04.json`:

```json
{
  "Clear Cell Renal Cell Carcinoma (ccRCC)": [
    "ccRCC Clear cytoplasm is the hallmark of this tumor, caused by dissolved lipids and glycogen.",
    "ccRCC A rich, delicate capillary network creates a chicken-wire appearance between tumor nests.",
    "ccRCC Tumor cells are organized in nested or alveolar architectures with sinusoidal spaces."
  ],
  "Papillary Renal Cell Carcinoma (pRCC)": [
    "pRCC Tumor growth is papillary with central fibrovascular cores lined by epithelial cells.",
    "pRCC Foamy macrophages are abundant in the papillary cores, forming a key diagnostic feature."
  ]
}
```

## Files Generated

- `extracted_content.json`: Raw extracted text and metadata
- `qa_pairs.json`: Generated question-answer pairs
- `tcga_style_descriptions.json`: Final TCGA-format descriptions
- `pipeline_report.txt`: Summary report of processing

## Example VQA Process

**Input Caption**: "Clear cell RCC shows clear cytoplasm."

**Generated QA Pair**:
- **Question**: "What is the cytoplasmic feature of clear cell RCC?"
- **Answer**: "The tumor cells have optically clear cytoplasm."

**Generated Description**: "ccRCC Clear cytoplasm is the hallmark of this tumor, caused by dissolved lipids and glycogen."

## Components

### PDFTextExtractor
- Uses LlamaParse for superior document understanding and text extraction
- Extracts relevant sections based on keywords
- Provides structured output with metadata
- Includes fallback extraction methods for robustness

### VQAAugmentationSystem
- Generates QA pairs using template-based approach
- Supports multiple question categories (morphology, differential, diagnostic, negative)
- Optional LLM enhancement for improved quality

### DescriptionGenerator
- Converts QA pairs to TCGA-style descriptions
- Maintains entity-specific formatting
- Handles deduplication and enhancement

## Requirements

- Python 3.8+
- LlamaParse for PDF processing (requires API key)
- OpenAI API key (optional, for LLM augmentation)

## License

This project is for educational and research purposes.