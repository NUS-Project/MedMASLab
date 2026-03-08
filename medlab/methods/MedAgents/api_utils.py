import base64
import os
import time
import random
import threading
from pathlib import Path
from methods.thread import qwen_generate_content
from openai import OpenAI

# 视频扩展名
_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# 媒体编码缓存：避免对同一图片/视频重复编码（尤其是视频抽帧）
_MEDIA_CACHE = {}
_MEDIA_CACHE_LOCK = threading.Lock()


def _get_media_parts(img_path_str):
    """获取单个媒体文件的 content parts，带缓存。"""
    if img_path_str in _MEDIA_CACHE:
        return _MEDIA_CACHE[img_path_str]

    with _MEDIA_CACHE_LOCK:
        if img_path_str in _MEDIA_CACHE:
            return _MEDIA_CACHE[img_path_str]

        parts = []
        ext = Path(img_path_str).suffix.lower().lstrip('.')
        if ext in _VIDEO_EXTENSIONS:
            from methods.vllm_thread import extract_frames
            frames = extract_frames(img_path_str)
            for frame_b64 in frames:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}})
        else:
            mime_map = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'webp': 'webp'}
            mime_type = mime_map.get(ext, 'jpeg')
            with open(img_path_str, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{encoded}"}})

        _MEDIA_CACHE[img_path_str] = parts
        return parts


def _build_multimodal_content(text_content, img_paths=None):
    """将文本内容转换为多模态内容格式（含图片/视频）。
    
    视频文件会自动抽帧为多张图片，结果会被缓存以避免重复抽帧。
    """
    if not img_paths:
        return text_content
    parts = []
    for img in img_paths:
        parts.extend(_get_media_parts(str(img)))
    parts.append({"type": "text", "text": text_content})
    return parts

# try:
#     from wrapt_timeout_decorator import timeout
# except Exception:  # pragma: no cover
#     def timeout(_seconds):
#         def _decorator(fn):
#             return fn
#
#         return _decorator

_CLIENT = None

# def _get_client() -> OpenAI:
#     global _CLIENT
#     if _CLIENT is not None:
#         return _CLIENT
# 
#     api_key = os.environ.get("OPENAI_API_KEY")
#     base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
#     if not api_key:
#         raise RuntimeError(
#             "Missing OPENAI_API_KEY for MedAgents. Set env vars or configure model_api_configs/model_api_config.json."
#         )
#     if not base_url:
#         raise RuntimeError(
#             "Missing OPENAI_BASE_URL for MedAgents. Set env vars or configure model_api_configs/model_api_config.json."
#         )
# 
#     _CLIENT = OpenAI(api_key=api_key, base_url=base_url)
#     return _CLIENT

# @timeout(600) # 600 seconds timeout (increased for large models like 72B)
def generate_response_multiagent(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input, img_paths=None,batch_manager=None):
    # print("Generating response for engine: ", engine)
    # start_time = time.time()
    # client = _get_client()
    # 如果有图片，注入到用户消息中
    user_content = _build_multimodal_content(user_input, img_paths) if img_paths else user_input
    response, prompt_tokens, completion_tokens = qwen_generate_content(messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_content},
        ], batch_manager=batch_manager)
    # response = client.chat.completions.create(
    #     model=engine,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=1,
    #     frequency_penalty=frequency_penalty,
    #     presence_penalty=presence_penalty,
    #     stop=stop,
    #     messages=[
    #         {"role": "system", "content": system_role},
    #         {"role": "user", "content": user_content},
    #     ],
    # )
    # end_time = time.time()
    # print('Finish!')
    # print("Time taken: ", end_time - start_time)

    return response,prompt_tokens, completion_tokens

# @timeout(600) # 600 seconds timeout (increased for large models like 72B)
def generate_response(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text, img_paths=None,batch_manager=None):
    print("Generating response for engine: ", engine)
    # start_time = time.time()
    # client = _get_client()
    # 如果有图片，注入到用户消息中
    user_content = _build_multimodal_content(input_text, img_paths) if img_paths else input_text
    response, prompt_tokens, completion_tokens = qwen_generate_content(messages=[{"role": "user", "content": user_content}], batch_manager=batch_manager)
    # response = client.chat.completions.create(
    #     model=engine,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=1,
    #     frequency_penalty=frequency_penalty,
    #     presence_penalty=presence_penalty,
    #     stop=stop,
    #     messages=[{"role": "user", "content": user_content}],
    # )
    # end_time = time.time()
    # print('Finish!')
    # print("Time taken: ", end_time - start_time)

    return response,prompt_tokens, completion_tokens

# @timeout(20) # 20 seconds timeout
# def generate_response_ins(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text, suffix, echo):
#     print("Generating response for engine: ", engine)
#     start_time = time.time()
#     # client = _get_client()
#     response = client.completions.create(
#         model=engine,
#         prompt=input_text,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         top_p=1,
#         suffix=suffix,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty,
#         stop=stop,
#         echo=echo,
#         logprobs=1,
#     )
#     end_time = time.time()
#     print('Finish!')
#     print("Time taken: ", end_time - start_time)
#
#     return response

class api_handler:
    def __init__(self, img_paths=None,model_info=None,batch_manager=None):
        # self.model = model
        self.img_paths = img_paths
        self.engine=model_info
        self.batch_manager=batch_manager


    def get_output_multiagent(self, system_role, user_input, max_tokens, temperature=0,
                    frequency_penalty=0, presence_penalty=0, stop=None):
        max_attempts = 3
        for attempt in range(max_attempts):
            # try:
            response ,prompt_tokens, completion_tokens= generate_response_multiagent(self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input, img_paths=self.img_paths,batch_manager=self.batch_manager)
            self._last_usage = { "prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens}
            return response
            # if response.choices and response.choices[0].message and getattr(response.choices[0].message, "content", None) is not None:
            #     return response.choices[0].message.content
            # else:
            #     return "ERROR."
            # except Exception as error:
            #     print(f'Attempt {attempt+1} of {max_attempts} failed with error: {error}')
            #     if attempt == max_attempts - 1:
            #         return "ERROR."


    def get_output(self, input_text, max_tokens, temperature=0,
                   suffix=None, stop=None, do_tunc=False, echo=False, ban_pronoun=False,
                   frequency_penalty=0, presence_penalty=0, return_prob=False):
        # try:
        response,prompt_tokens, completion_tokens = generate_response(self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text, img_paths=self.img_paths,batch_manager=self.batch_manager)
        # except Exception:
        #     print("Timeout")
        #     try:
        #         response = generate_response(self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text, img_paths=self.img_paths)
        #     except Exception:
        #         print("Timeout occurred again. Exiting.")
        #         response = "ERROR."
        #         return response  # 直接返回空字符串
        self._last_usage = { "prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens}
        # if (hasattr(response, 'choices') and response.choices
        #         and response.choices[0].message
        #         and getattr(response.choices[0].message, "content", None) is not None):
        x = response
        # else:
        #     # print(response)
        #     x = "ERROR."
        #     return x

        if do_tunc:
            y = x.strip() # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            if '\n' in y:
                pos = y.find('\n') # 这里的意思是找到第一个换行符的位置
                y = y[:pos] # 这里的意思是把第一个换行符之前的内容保留
            if 'Q:' in y:
                pos = y.find('Q:')
                y = y[:pos]
            if 'Question:' in y:
                pos = y.find('Question:')
                y = y[:pos]
            assert not ('\n' in y)
            if not return_prob:
                return y

        if not return_prob:
            return x

        # pdb.set_trace()
        output_token_offset_real, output_token_tokens_real, output_token_probs_real = [], [], []
        return x, (output_token_offset_real, output_token_tokens_real, output_token_probs_real)

"""
(Pdb) x
' Academy Award because The Curious Case of Benjamin Button won three Academy Awards, which are given by the Academy of Motion Picture Arts and Sciences.'
(Pdb) output_token_offset_real
[0, 8, 14, 22, 26, 34, 39, 42, 51, 58, 62, 68, 76, 83, 84, 90, 94, 100, 103, 107, 115, 118, 125, 133, 138, 142, 151]
(Pdb) output_token_tokens_real
[' Academy', ' Award', ' because', ' The', ' Curious', ' Case', ' of', ' Benjamin', ' Button', ' won', ' three', ' Academy', ' Awards', ',', ' which', ' are', ' given', ' by', ' the', ' Academy', ' of', ' Motion', ' Picture', ' Arts', ' and', ' Sciences', '.']
(Pdb) output_token_probs_real
[-0.7266144, -0.68505085, -0.044669915, -0.00023392851, -0.0021017971, -2.1768952e-05, -1.1430258e-06, -6.827632e-08, -3.01145e-05, -1.2231317e-05, -0.07086051, -2.7967804e-05, -6.6619094e-07, -0.41155097, -0.0020535963, -0.0021325003, -0.6671403, -0.51776046, -0.00014945272, -0.41470888, -3.076318e-07, -3.583558e-05, -2.9311614e-06, -3.869565e-05, -1.1430258e-06, -9.606849e-06, -0.017712338]
"""

