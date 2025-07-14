#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

from pdf_extractor import PDFTextExtractor
from vqa_augmentation import VQAAugmentationSystem
from description_generator import DescriptionGenerator

class PathologyPipeline:
    """
    Main pipeline for processing pathology PDFs into TCGA-style descriptions
    using VQA-based augmentation as reasoning step.
    """
    
    def __init__(self, config: Dict[str, Any] = None, llama_api_key: Optional[str] = None):
        self.config = config or self._default_config()
        
        # Initialize components with API key
        self.llama_api_key = llama_api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        self.pdf_extractor = PDFTextExtractor(api_key=self.llama_api_key)
        self.vqa_system = VQAAugmentationSystem()
        self.description_generator = DescriptionGenerator(llama_api_key=self.llama_api_key)
        
        self.output_dir = Path(self.config.get("output_dir", "output"))
        self.output_dir.mkdir(exist_ok=True)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the pipeline"""
        return {
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
            "save_intermediate": True,
            "use_llm_augmentation": False  # Set to True if OpenAI API key available
        }
    
    def run_pipeline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Run the complete pipeline:
        1. Extract text from PDF
        2. Generate VQA pairs as reasoning step
        3. Generate TCGA-style descriptions
        """
        results = {}
        
        print(f"Processing PDF: {pdf_path}")
        
        # Step 1: Extract text from PDF
        print("Step 1: Extracting text from PDF...")
        extracted_content = self.pdf_extractor.process_pathology_pdf(pdf_path)
        
        if self.config["save_intermediate"]:
            with open(self.output_dir / "extracted_content.json", "w") as f:
                json.dump(extracted_content, f, indent=2)
        
        results["extracted_content"] = extracted_content
        
        # Step 2: Generate VQA pairs (reasoning step)
        print("Step 2: Generating VQA pairs for reasoning...")
        qa_pairs = self.vqa_system.process_extracted_content(extracted_content)
        
        if self.config["save_intermediate"]:
            qa_output = {}
            for entity, pairs in qa_pairs.items():
                qa_output[entity] = [
                    {
                        "question": pair.question,
                        "answer": pair.answer,
                        "context": pair.context,
                        "category": pair.category
                    }
                    for pair in pairs
                ]
            
            with open(self.output_dir / "qa_pairs.json", "w") as f:
                json.dump(qa_output, f, indent=2)
        
        results["qa_pairs"] = qa_pairs
        
        # Step 3: Generate TCGA-style descriptions
        print("Step 3: Generating TCGA-style descriptions...")
        descriptions = self.description_generator.generate_tcga_style_descriptions(qa_pairs)
        
        # Save final descriptions
        output_path = self.output_dir / "tcga_style_descriptions.json"
        self.description_generator.save_descriptions(descriptions, str(output_path))
        
        results["descriptions"] = descriptions
        
        print(f"Pipeline completed. Results saved to: {self.output_dir}")
        
        return results
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create a summary report of the pipeline results"""
        report = []
        report.append("=== Pathology PDF Processing Pipeline Report ===\n")
        
        # Extraction summary
        extracted = results.get("extracted_content", {})
        metadata = extracted.get("metadata", {})
        
        report.append(f"PDF Processing Summary:")
        report.append(f"  - Total pages: {metadata.get('total_pages', 'N/A')}")
        report.append(f"  - Relevant sections found: {len(extracted.get('relevant_sections', {}))}")
        
        # QA pairs summary
        qa_pairs = results.get("qa_pairs", {})
        report.append(f"\nVQA Reasoning Summary:")
        total_qa = sum(len(pairs) for pairs in qa_pairs.values())
        report.append(f"  - Total QA pairs generated: {total_qa}")
        
        for entity, pairs in qa_pairs.items():
            categories = {}
            for pair in pairs:
                categories[pair.category] = categories.get(pair.category, 0) + 1
            
            report.append(f"  - {entity}: {len(pairs)} pairs")
            for category, count in categories.items():
                report.append(f"    * {category}: {count}")
        
        # Descriptions summary
        descriptions = results.get("descriptions", {})
        report.append(f"\nTCGA-Style Descriptions Generated:")
        total_descriptions = sum(len(desc_list) for desc_list in descriptions.values())
        report.append(f"  - Total descriptions: {total_descriptions}")
        
        for entity, desc_list in descriptions.items():
            report.append(f"  - {entity}: {len(desc_list)} descriptions")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Process pathology PDF to TCGA-style descriptions")
    parser.add_argument("pdf_path", help="Path to the pathology PDF file")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--llama-api-key", help="LlamaCloud API key (can also use LLAMA_CLOUD_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check for API key
    llama_api_key = args.llama_api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_api_key:
        print("Error: LLAMA_CLOUD_API_KEY is required.")
        print("Set it as environment variable or use --llama-api-key argument")
        return 1
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    
    if not config:
        config = {}
    
    config["output_dir"] = args.output_dir
    
    # Run pipeline
    pipeline = PathologyPipeline(config, llama_api_key=llama_api_key)
    
    try:
        results = pipeline.run_pipeline(args.pdf_path)
        
        # Generate and save report
        report = pipeline.create_summary_report(results)
        
        with open(Path(args.output_dir) / "pipeline_report.txt", "w") as f:
            f.write(report)
        
        print("\n" + report)
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())