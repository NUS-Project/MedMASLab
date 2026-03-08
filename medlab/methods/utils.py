import os
import json
from pathlib import Path
import base64
import yaml
import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText,LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import time
from qwen_vl_utils import process_vision_info


_MODEL_CACHE = {}
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def handle_retry_error(retry_state):
    print(f"Retry failed")

def is_options():
    # if need_judge:
    return "Based on the above information, come up with your own concise and accurate final opinion. Provide only your final answer.No other Comments like your thinking and reasoning process!"
    # else:
    #     return "Provide only the letter corresponding to your answer choice (A/B/C/D/E/F),No other Comments like your thinking and reasoning process! Your final answer should follow this format strictly: \nAnswer: <your answer>.\nFor Example:Answer: A or Answer: B or Answer: C or Answer: D or Answer: E or Answer: F."

def get_gpu_memory_usage():
    # if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    # reserved_memory = torch.cuda.memory_reserved(0)
    usage_percentage = (allocated_memory / total_memory) * 100
    print(f"GPU Memory Usage: {allocated_memory / (1024**2):.2f} MB / {total_memory / (1024**2):.2f} MB ({usage_percentage:.2f}%)")
    return allocated_memory, total_memory, usage_percentage
    # else:
    #     print("No CUDA available.")
    #     return None


def _load_json_if_exists(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chat_content(image_path,message):
    """Encodes the image at the given path to a base64 string."""
    base64_images = []
    # if image_path is None:
    #     return None
    for img in image_path:
        base64_images.append(encode_image(img))
    content = [{"type": "text", "text": message}]
    for base64_image in base64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    result={"role": "user", "content": content}
    return result


def setup_model(model_name,root_path):
    # root_path="/mnt/dhwfile/raise/user/panjiabao/huxiaobin/yunhang/"
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    # model_path=os.getenv("GENERAL_MODEL_PATH")
    if model_name == 'Llama-3.3-70B-Instruct':
        model_path=Path(root_path) / 'models' / 'Llama-3.3-70B-Instruct'
        pipeline = transformers.pipeline(
            "text-generation",
            model=os.path.join(model_path, model_name),
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        _MODEL_CACHE[model_name] = pipeline
        return pipeline
   
    elif model_name=='Qwen2.5-VL-7B-Instruct':
        
        model_path=Path(root_path) / 'models' / 'Qwen2.5-VL-7B-Instruct'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        _MODEL_CACHE[model_name] =  (model, processor)
        # _MODEL_CACHE[model_name] =model
        return _MODEL_CACHE[model_name]
        
    elif model_name=='Qwen2.5-VL-32B-Instruct':
        model_path=Path(root_path) / 'models' / 'Qwen2.5-VL-32B-Instruct'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        _MODEL_CACHE[model_name] =  (model, processor)
        # _MODEL_CACHE[model_name] =model
        return _MODEL_CACHE[model_name]
        
    
    elif model_name=='Qwen2.5-VL-72B-Instruct':
        model_path=Path(root_path) / 'models' / 'Qwen2.5-VL-72B-Instruct'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        _MODEL_CACHE[model_name] =  (model, processor)
        # _MODEL_CACHE[model_name] =model
        return _MODEL_CACHE[model_name]

    elif model_name=='LLaVA-NeXT-Video-7B-hf':
        
        model_path=Path(root_path) / 'models' / 'LLaVA-NeXT-Video-7B-hf'
        model= AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        processor =  LlavaNextVideoProcessor.from_pretrained(model_path)
        _MODEL_CACHE[model_name] =  (model, processor)
        # _MODEL_CACHE[model_name] =model
        return _MODEL_CACHE[model_name]

    elif model_name=='LLaVA-NeXT-Video-34B-hf':
        
        model_path=Path(root_path) / 'models' / 'LLaVA-NeXT-Video-34B-hf'
        model= LlavaNextVideoForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        processor = AutoProcessor.from_pretrained(model_path)
        _MODEL_CACHE[model_name] =  (model, processor)
        # _MODEL_CACHE[model_name] =model
        return _MODEL_CACHE[model_name]
    # else:
    #     raise ValueError(f"Unsupported model: {model_name}")

def qwen_vl_chat_content(image_path=None,message=None):
   
    content = [{"type": "text", "text": message}]
    if image_path is not None:
        for img in image_path:
            img=str(img)
            if "mp4" not in str(img):
                content.append({"type": "image", "image": img})
            else:
                content.append({"type": "video", "video": img})

    result={"role": "user", "content": content}
    return result


def qwen_vl_generate_content(model,processor,messages):
    has_video = False
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "video":
                        has_video = True
                        # print(f"Found video: {item.get('video')}")
                        break
    text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    if has_video:
        # process_vision_info time:1.3101625442504883                                                             
                                                                                                        
        # processor time:0.5336735248565674                                                                       
                                                                                                                
        # iputs to cuda time:0.13800358772277832                                                                  
                                                                                                                
        # generated time:42.48481726646423     
        start_time = time.time()
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        end_time = time.time()
        print(f"\nprocess_vision_info time:{end_time-start_time}")


        start_time = time.time()
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # fps=30.0,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        end_time = time.time()
        print(f"\nprocessor time:{end_time-start_time}")

    else:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
    start_time = time.time()
    inputs = inputs.to(model.device)
    end_time = time.time()
    print(f"\niputs to cuda time:{end_time-start_time}")


    # Generate output from the model
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=51200,
            do_sample=True,
            temperature=0.1
        )
    end_time = time.time()
    print(f"\ngenerated time:{end_time-start_time}")

    # Decode the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0],inputs.input_ids.shape[1],generated_ids_trimmed[0].shape[0]


def encode_image(image_path):
    """Encodes the image at the given path to a base64 string."""
    if image_path is None:
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_apikey_and_baseurl_from_configs(root_path=None, model_info=None):
    """Populate OPENAI_API_KEY / BASE_URL from repo configs if env vars are missing.

    Precedence:
    1) Existing env vars
    2) model_api_configs/model_api_config.json entry for model_info
    """

    resolved_root = str(Path(root_path).expanduser().resolve())
    api_cfg_path = Path(resolved_root) / 'model_api_configs' / 'model_api_config.json'
    api_cfg = _load_json_if_exists(str(api_cfg_path))
    model_cfg = api_cfg.get(model_info, {}) if isinstance(api_cfg, dict) else {}
    api_key=model_cfg['api_key']
    url_from_cfg = model_cfg.get('base_url')
    if not model_cfg:
        raise RuntimeError(
            f"Missing model config for '{model_info}' in {api_cfg_path}. "
            "Add an entry with at least 'api_key' and 'model_url' (or 'base_url')."
        )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it via env var or 'api_key' in model_api_configs/model_api_config.json."
        )
    if not url_from_cfg:
        raise RuntimeError(
            "BASE_URL is not set. Provide it via env var or 'model_url'/'base_url' in model_api_configs/model_api_config.json."
        )
    return api_key, url_from_cfg
