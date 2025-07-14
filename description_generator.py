from typing import Dict, List, Any, Optional
import json
import re
import os
from dataclasses import dataclass
from pdf_extractor import PDFTextExtractor
from vqa_augmentation import VQAAugmentationSystem, QAPair

@dataclass
class PathologyDescription:
    entity: str
    descriptions: List[str]
    
class DescriptionGenerator:
    def __init__(self, llama_api_key: Optional[str] = None):
        self.pdf_extractor = PDFTextExtractor(api_key=llama_api_key)
        self.vqa_system = VQAAugmentationSystem()
        
        # Template patterns for generating descriptions
        self.description_templates = {
            "morphology": "{entity} {feature} {description}.",
            "differential": "{entity} {comparison} {other_entity}.",
            "negative": "{entity} does not {negative_feature}.",
            "positive": "{entity} {positive_feature}."
        }
    
    def generate_tcga_style_descriptions(self, qa_pairs: Dict[str, List[QAPair]]) -> Dict[str, List[str]]:
        """Generate TCGA-style descriptions from QA pairs"""
        descriptions = {}
        
        for entity, pairs in qa_pairs.items():
            entity_descriptions = []
            
            for pair in pairs:
                # Convert QA pair to description format
                description = self._qa_to_description(pair, entity)
                if description:
                    entity_descriptions.append(description)
            
            # Remove duplicates and sort
            unique_descriptions = list(set(entity_descriptions))
            descriptions[entity] = unique_descriptions
        
        return descriptions
    
    def _qa_to_description(self, qa_pair: QAPair, entity: str) -> str:
        """Convert QA pair to TCGA-style description"""
        entity_code = self._get_entity_code(entity)
        
        # Extract key information from answer
        answer = qa_pair.answer.lower()
        
        # Template-based description generation
        if "cytoplasm" in qa_pair.question.lower():
            if "clear" in answer:
                return f"{entity_code} Clear cytoplasm is the hallmark of this tumor, caused by dissolved lipids and glycogen."
            elif "eosinophilic" in answer:
                return f"{entity_code} Tumor cells have pale eosinophilic cytoplasm filled with microvesicles."
        
        elif "nuclear" in qa_pair.question.lower() or "nuclei" in qa_pair.question.lower():
            if "round" in answer:
                return f"{entity_code} Low to intermediate-grade nuclei with round contours are common."
            elif "wrinkled" in answer:
                return f"{entity_code} The nuclei appear wrinkled and irregular — often described as raisinoid."
        
        elif "architecture" in qa_pair.question.lower():
            if "nested" in answer:
                return f"{entity_code} Tumor cells are organized in nested or alveolar architectures."
            elif "papillary" in answer:
                return f"{entity_code} Tumor growth is papillary with central fibrovascular cores."
            elif "solid" in answer:
                return f"{entity_code} Forms solid sheets or trabecular cords of cells."
        
        elif "differ" in qa_pair.question.lower() or "distinguish" in qa_pair.question.lower():
            return f"{entity_code} {qa_pair.answer}"
        
        elif "not" in qa_pair.question.lower() or "lack" in qa_pair.question.lower():
            return f"{entity_code} {qa_pair.answer}"
        
        # Generic description format
        return f"{entity_code} {qa_pair.answer}"
    
    def _get_entity_code(self, entity: str) -> str:
        """Get entity code for description prefix"""
        if "clear cell" in entity.lower():
            return "ccRCC"
        elif "papillary" in entity.lower():
            return "pRCC"
        elif "chromophobe" in entity.lower():
            return "chRCC"
        return entity
    
    def enhance_descriptions_with_reasoning(self, descriptions: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Enhance descriptions using reasoning from VQA process"""
        enhanced_descriptions = {}
        
        for entity, desc_list in descriptions.items():
            enhanced_list = []
            
            for desc in desc_list:
                # Add reasoning context
                enhanced_desc = self._add_reasoning_context(desc, entity)
                enhanced_list.append(enhanced_desc)
            
            enhanced_descriptions[entity] = enhanced_list
        
        return enhanced_descriptions
    
    def _add_reasoning_context(self, description: str, entity: str) -> str:
        """Add reasoning context to description"""
        # This would be enhanced with actual reasoning from the VQA process
        # For now, return the original description
        return description
    
    def process_pdf_to_descriptions(self, pdf_path: str) -> Dict[str, List[str]]:
        """Complete pipeline: PDF -> QA -> Descriptions"""
        # Step 1: Extract text from PDF
        extracted_content = self.pdf_extractor.process_pathology_pdf(pdf_path)
        
        # Step 2: Generate QA pairs
        qa_pairs = self.vqa_system.process_extracted_content(extracted_content)
        
        # Step 3: Generate descriptions
        descriptions = self.generate_tcga_style_descriptions(qa_pairs)
        
        # Step 4: Enhance with reasoning
        enhanced_descriptions = self.enhance_descriptions_with_reasoning(descriptions)
        
        return enhanced_descriptions
    
    def save_descriptions(self, descriptions: Dict[str, List[str]], output_path: str):
        """Save descriptions in TCGA format"""
        # Format similar to TCGA_RCC_desc_04.json
        formatted_descriptions = {}
        
        for entity, desc_list in descriptions.items():
            # Create full entity name
            if entity == "Clear Cell RCC":
                key = "Clear Cell Renal Cell Carcinoma (ccRCC)"
            elif entity == "Papillary RCC":
                key = "Papillary Renal Cell Carcinoma (pRCC)"
            elif entity == "Chromophobe RCC":
                key = "Chromophobe Renal Cell Carcinoma (chRCC)"
            else:
                key = entity
            
            formatted_descriptions[key] = desc_list
        
        with open(output_path, "w") as f:
            json.dump(formatted_descriptions, f, indent=2)
    
    def generate_sample_descriptions(self) -> Dict[str, List[str]]:
        """Generate sample descriptions for testing"""
        sample_descriptions = {
            "Clear Cell Renal Cell Carcinoma (ccRCC)": [
                "ccRCC Clear cytoplasm is the hallmark of this tumor, caused by dissolved lipids and glycogen.",
                "ccRCC A rich, delicate capillary network creates a chicken-wire appearance between tumor nests.",
                "ccRCC Tumor cells are organized in nested or alveolar architectures with sinusoidal spaces.",
                "ccRCC Low to intermediate-grade nuclei with round contours and small nucleoli are common in ccRCC.",
                "ccRCC This tumor lacks papillary structures with fibrovascular cores, which are common in pRCC.",
                "ccRCC Unlike chRCC, ccRCC does not have perinuclear halos or a plant-cell appearance."
            ],
            "Papillary Renal Cell Carcinoma (pRCC)": [
                "pRCC Tumor growth is papillary with central fibrovascular cores lined by epithelial cells.",
                "pRCC Foamy macrophages are abundant in the papillary cores, forming a key diagnostic feature.",
                "pRCC Psammoma bodies, or concentrically laminated calcifications, are commonly seen.",
                "pRCC The tumor often presents as multiple discrete nodules (multifocality) within the kidney.",
                "pRCC pRCC does not exhibit the clear cytoplasm seen in ccRCC due to lack of lipid/glycogen clearing.",
                "pRCC Unlike chRCC, it lacks well-defined cell borders and perinuclear halos."
            ],
            "Chromophobe Renal Cell Carcinoma (chRCC)": [
                "chRCC Tumor cells have pale eosinophilic cytoplasm filled with microvesicles under H&E staining.",
                "chRCC Perinuclear halos are a defining cytologic feature, creating a clearing zone around nuclei.",
                "chRCC The nuclei appear wrinkled and irregular — often described as raisinoid.",
                "chRCC Forms solid sheets or trabecular cords of cells with minimal architectural complexity.",
                "chRCC Tumor cells display well-defined cell membranes, giving a plant-cell or cobblestone appearance.",
                "chRCC chRCC does not contain clear cytoplasm or lipid-rich vacuoles like ccRCC."
            ]
        }
        
        return sample_descriptions

if __name__ == "__main__":
    generator = DescriptionGenerator()
    
    # Test with sample descriptions
    sample_descriptions = generator.generate_sample_descriptions()
    generator.save_descriptions(sample_descriptions, "generated_descriptions.json")
    
    print("Generated sample descriptions in TCGA format")
    
    # If PDF exists, process it
    try:
        pdf_descriptions = generator.process_pdf_to_descriptions("Pathoma 2021.pdf")
        generator.save_descriptions(pdf_descriptions, "pdf_generated_descriptions.json")
        print("Generated descriptions from PDF")
    except Exception as e:
        print(f"Could not process PDF: {e}")