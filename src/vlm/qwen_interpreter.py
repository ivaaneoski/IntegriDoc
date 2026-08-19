import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.vlm.prompt_builder import build_forensic_prompt

class ForensicVLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = None):
        """
        Initializes the Qwen2.5-VL model for forensic explanation.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading VLM {model_name} on {self.device}...")
        
        # Load the model on the available device with appropriate dtype
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # CPU fallback if auto device_map doesn't work out
        if self.device == "cpu":
            self.model = self.model.to("cpu")
            
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def analyze_document(self, image_path: str, ocr_data: list, classifier_score: float, has_heatmap: bool = True):
        """
        Analyzes a document image along with its extracted evidence.
        """
        # 1. Build the prompt
        prompt_text = build_forensic_prompt(ocr_data, classifier_score, has_heatmap)
        
        # 2. Format the message for Qwen-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        # 3. Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # 4. Generate
        print("Generating forensic report...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.1, # Keep it deterministic and factual
                do_sample=False
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 5. Extract JSON (VLM might wrap it in ```json ... ```)
        cleaned_output = output_text.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]
            
        try:
            return json.loads(cleaned_output.strip())
        except json.JSONDecodeError:
            print("Failed to parse JSON. Raw output:")
            print(output_text)
            return {"error": "Invalid JSON generated", "raw_output": output_text}
