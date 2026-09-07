# -*- coding: utf-8 -*-
"""
Reasoning Stratification (L1/L2/L3) & Open-Ended Prompt Transformer
Strips MCQ choices and builds multi-tier clinical instructions
"""
import re

class PromptStratificationEngine:
    def __init__(self):
        self.mcq_pattern = re.compile(r'\b[A-D]\)\s*[^\n]+', re.IGNORECASE)
        
    def transform_mcq_to_open_ended(self, raw_question, options_text=""):
        cleaned = self.mcq_pattern.sub("", raw_question).strip()
        cleaned = re.sub(r'\s*Which of the following[^?]*\?', '', cleaned, flags=re.IGNORECASE)
        if not cleaned.endswith("?"):
            cleaned = "Based on the chest radiograph, " + cleaned
            if not cleaned.endswith("."):
                cleaned += "."
        return cleaned

    def classify_tier(self, question, context=""):
        q_low = (question + " " + context).lower()
        if any(k in q_low for k in ["etiology", "cause", "underlying condition", "differentials", "diagnosis", "failure"]):
            return "L3: Complex Diagnosis"
        elif any(k in q_low for k in ["where", "lobe", "zone", "location", "bilateral", "left", "right", "apex", "base"]):
            return "L2: Localization"
        else:
            return "L1: Recognition"

    def format_qwen3_vl_prompt(self, image_path, question):
        tier = self.classify_tier(question)
        open_q = self.transform_mcq_to_open_ended(question)
        system_msg = (
            "You are an expert thoracic radiologist. Analyze the provided chest radiograph carefully. "
            "Formulate a concise, clinically accurate findings report and diagnosis using standard RadLex terminology."
        )
        tier_instruction = {
            "L1: Recognition": "Assess radiographic technique, projection, and overall normalcy.",
            "L2: Localization": "Specify precise anatomic lobes, lung zones, mediastinal, or pleural spaces.",
            "L3: Complex Diagnosis": "Synthesize radiological signs into primary etiologies and differential diagnoses."
        }[tier]
        
        return {
            "system": system_msg,
            "tier": tier,
            "instruction": tier_instruction,
            "transformed_prompt": f"<|im_start|>system\n{system_msg}\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{open_q} {tier_instruction}<|im_end|>\n<|im_start|>assistant\n"
        }

if __name__ == "__main__":
    engine = PromptStratificationEngine()
    sample_mcq = "What is the primary abnormality seen in the right lower lobe? A) Pneumonia B) Normal C) Pneumothorax D) Rib Fracture"
    result = engine.format_qwen3_vl_prompt("sample_cxr.png", sample_mcq)
    print("=" * 70)
    print("PROMPT STRATIFICATION & TRANSFORMATION DEMO")
    print("=" * 70)
    print(f"Original MCQ      : {sample_mcq}")
    print(f"Classified Tier   : {result['tier']}")
    print(f"Tier Instruction  : {result['instruction']}")
    print(f"Engine Prompt     :\n{result['transformed_prompt']}")
    print("=" * 70)
