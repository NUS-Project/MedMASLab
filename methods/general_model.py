import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoModelForCausalLM, AutoTokenizer

from qwen_vl_utils import process_vision_info
from methods.utils import qwen_vl_chat_content,is_options,setup_model,qwen_vl_generate_content
from methods.thread import qwen_generate_content


class BaseLine_Test:
    def __init__(self, batch_manager):
        self.batch_manager=batch_manager

    def chat(self, description,image_path):
        # if need_judge:
        msg=f"You are a medical expert.{is_options()}"
      
        # msg="You are a medical expert."
        messages = [
            {
                "role": "system",
                 "content": [{"type": "text", "text":msg }]
        
            }
        ]

        messages.append(qwen_vl_chat_content(image_path,description+is_options()))
        completion,prompt_tokens,completion_tokens=qwen_generate_content(messages,self.batch_manager)
            

        token_stats = {
            "BaseLine": {"num_llm_calls": 1, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        }
        current_config = {"current_num_agents": 1, "round": 1}

        return completion,token_stats,current_config

def test_BaseLine(description,image_path,batch_manager):
    BaseLine=BaseLine_Test(batch_manager)
    completion,token_stats,current_config=BaseLine.chat(description,image_path)
    return completion,token_stats,current_config

class Llama_test:
    def __init__(self, model_path: str):
        """
        Initializes the model and the text generation pipeline.

        Args:
            model_path (str): The path to the pretrained model.
        """
        self.model_path = model_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path)  # Initialize the tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def chat(self, user_input: str):
        """
        Generates a response from the chatbot based on user input.

        Args:
            user_input (str): The message from the user.

        Returns:
            str: The chatbot's response.
        """
        # Prepare messages for the model input
        messages = [
            {"role": "system",
             "content": "You are a medical expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."},
            {"role": "user", "content": user_input},
        ]

        # Combine messages into a single string for tokenization
        prompt = ' '.join([msg['content'] for msg in messages])

        # Calculate prompt tokens
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]  # Token count

        # Generate output using the pipeline
        outputs = self.pipeline(
            prompt,
            max_new_tokens=2560,
        )

        # Calculate completion tokens
        completion_tokens = len(outputs[0]["generated_text"].split())

        # Update token statistics
        token_stats = {
            "Llama": {
                "num_llm_calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
        }

        # Extract and return the latest response
        return outputs[0]["generated_text"],token_stats
