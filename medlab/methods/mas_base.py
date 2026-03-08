import base64
import os
import random
import openai
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
from methods.thread import qwen_generate_content
from methods.utils import handle_retry_error, load_config

_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}


def _encode_image_to_base64(image_path) -> str:
    """将图片文件编码为 base64 data URL，供 VLM API 使用。"""
    image_path = str(image_path)
    ext = Path(image_path).suffix.lower().lstrip('.')
    mime_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}
    mime_type = mime_map.get(ext, 'jpeg')
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/{mime_type};base64,{encoded}"


def _encode_media_to_content_parts(file_path):
    """将图片或视频文件编码为 OpenAI 多模态内容列表。

    图片返回 1 个 image_url 条目；视频抽帧后返回多个 image_url 条目。
    """
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower().lstrip('.')
    if ext in _VIDEO_EXTENSIONS:
        from methods.vllm_thread import extract_frames
        frames = extract_frames(file_path)
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames]
    else:
        data_url = _encode_image_to_base64(file_path)
        return [{"type": "image_url", "image_url": {"url": data_url}}]


def _inject_images_into_messages(messages, img_paths):
    """将图片/视频注入到消息列表的第一条 user 消息中（多模态格式）。

    修改 messages in-place：将第一条 user 消息的 content 从字符串
    转换为 OpenAI 多模态内容列表 [{"type": "image_url", ...}, {"type": "text", ...}]。
    视频文件会自动抽帧为多张图片。
    """
    if not img_paths:
        return messages
    for msg in messages:
        if msg.get("role") == "user":
            text_content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            content_parts = []
            for img in img_paths:
                content_parts.extend(_encode_media_to_content_parts(img))
            content_parts.append({"type": "text", "text": text_content})
            msg["content"] = content_parts
            break  # 只修改第一条 user 消息
    return messages


class MAS():

    def __init__(self, model_name=None, batch_manager=None):
        self.batch_manager = batch_manager
        self.model_name=model_name

        if model_name is not None:
            # Get the child class's module path
            child_module_path = os.path.dirname(os.path.abspath(self.__class__.__module__.replace('.', '/')))
            self.method_config = load_config(os.path.join(child_module_path, "configs", "config_main.yaml"))

        # self.model_api_config = general_config["model_api_config"]
        # self.model_name = general_config["model_name"]
        # self.model_temperature = general_config["model_temperature"]
        # self.model_max_tokens = general_config["model_max_tokens"]
        # self.model_timeout = general_config["model_timeout"]

        # Tracking compute costs
        self.token_stats = {
            self.model_name: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }

        self.memory_bank = {}
        self.tools = {}

    def inference(self, sample):
        """
        sample: data sample (dictionary) to be passed to the MAS
        """
        query = sample["query"]
        response = self.call_llm(prompt=query)
        return {"response": response}

    # @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt=None, system_prompt=None, messages=None, model_name=None, temperature=None,
                 img_paths=None):

        # model_name = model_name if model_name is not None else self.model_name
        # model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        # model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']

        if messages is None:
            assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]

        # 如果有图片，注入到消息中（多模态支持）
        if img_paths:
            messages = _inject_images_into_messages(messages, img_paths)

        # model_temperature = temperature if temperature is not None else self.model_temperature
        # print(
        #     f"\nmessages:{[{k: (str(v)[:100] + '...' if isinstance(v, (str, list)) and len(str(v)) > 100 else v) for k, v in m.items()} for m in messages]}")

        # request_dict = {
        #     "model": model_name,
        #     "messages": messages,
        #     "max_tokens": self.model_max_tokens,
        #     "timeout": self.model_timeout
        # }
        # # if "o1" not in model_name:              # OpenAI's o1 models do not support temperature
        # #     request_dict["temperature"] = model_temperature

        # llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        # try:
        #     completion = llm.chat.completions.create(**request_dict)
        #     if not completion.choices or not completion.choices[0].message:
        #         raise ValueError("Empty choices returned from LLM API")
        #     response = completion.choices[0].message.content or ""
        #     usage = getattr(completion, "usage", None)
        #     num_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        #     num_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        # finally:
        #     llm.close()

        response, num_prompt_tokens, num_completion_tokens = qwen_generate_content(messages, self.batch_manager)

        if isinstance(response, str):
            if model_name not in self.token_stats:
                self.token_stats[model_name] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            self.token_stats[model_name]["num_llm_calls"] += 1
            self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
            self.token_stats[model_name]["completion_tokens"] += num_completion_tokens
        else:
            raise ValueError(f"Invalid response from LLM: {response}")

        return response

    def get_token_stats(self):
        return self.token_stats

    def optimizing(self, val_data):
        """
        For methods that requires validation data such as GPTSwarm and ADAS
        """
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass

    def get_tool(self):
        pass
