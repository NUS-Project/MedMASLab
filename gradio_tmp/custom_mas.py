from openai import OpenAI
import base64
from pathlib import Path

class MedicalAgent:
    def __init__(self, api_key, base_url, model_name, prompt):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.prompt = prompt

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def _get_image_media_type(self, image_path):
        extension = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(extension, "image/jpeg")

    def analyze_case(self, question, image_path=None):
        # Build message content
        content = [{"role": "user", "content": self.prompt + " " + question}]

        # Add image if provided
        if image_path:
            try:
                image_data = self._encode_image(image_path)
                media_type = self._get_image_media_type(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_data}"
                    }
                })
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_path}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=content,
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048
        )

        return response.choices[0].message.content


class PhysicianAgent(MedicalAgent):
    def __init__(self, api_key, base_url, model_name):
        prompt = (
            "You are an experienced Internal Medicine physician. Analyze the patient case thoroughly, "
            "combining clinical evidence with your expertise. Provide a comprehensive differential diagnosis "
            "considering the patient's demographics, symptoms, and test results. "
            "Always prioritize the most common and dangerous diagnoses first."
        )
        super().__init__(api_key, base_url, model_name, prompt)


class NeurologistAgent(MedicalAgent):
    def __init__(self, api_key, base_url, model_name):
        prompt = (
            "You are a specialized neurologist. For neurological cases, analyze symptoms, reflex patterns, "
            "and imaging findings. Consider central vs peripheral nervous system involvement. Assess for stroke, "
            "seizure, dementia, or other neurological conditions with expert precision."
        )
        super().__init__(api_key, base_url, model_name, prompt)


class RadiologistAgent(MedicalAgent):
    def __init__(self, api_key, base_url, model_name):
        prompt = (
            "You are an expert radiologist. Interpret imaging findings carefully, identifying pathological abnormalities. "
            "Describe your observations in medical terminology, assess diagnostic certainty, and suggest follow-up imaging "
            "if needed. Correlate imaging with clinical presentation."
        )
        super().__init__(api_key, base_url, model_name, prompt)


def test_sample(question, image_path=None):
    api_key = "sk-pqQMGWFk3QDIwDubCC4AAGVh1JigOAh1d6yowtZdzQZmttiC"
    base_url = "https://api.vectorengine.ai/v1"
    model_name = "gpt-4o-mini"

    physician_agent = PhysicianAgent(api_key, base_url, model_name)
    neurologist_agent = NeurologistAgent(api_key, base_url, model_name)
    radiologist_agent = RadiologistAgent(api_key, base_url, model_name)

    # Collect responses from all agents
    physician_answer = physician_agent.analyze_case(question, image_path)
    neurologist_answer = neurologist_agent.analyze_case(question, image_path)
    radiologist_answer = radiologist_agent.analyze_case(question, image_path)

    # Combine answers (this is a simple approach; you may want to implement a more sophisticated method)
    final_answer = f"Physician's Analysis: {physician_answer}\n" \
                   f"Neurologist's Analysis: {neurologist_answer}\n" \
                   f"Radiologist's Analysis: {radiologist_answer}"

    return final_answer