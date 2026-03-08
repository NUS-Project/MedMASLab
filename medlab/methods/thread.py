import threading
import queue,time,torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
import uuid
from PIL import Image
from pathlib import Path
from transformers import pipeline
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from methods.utils import setup_model
import av
import numpy as np
import copy
from copy import deepcopy 

@dataclass
class InferenceRequest:
    """推理请求数据结构"""
    request_id: str
    messages: List[Dict]
    result_queue: queue.Queue  # 用于返回结果的队列



def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def filter_duplicate_images(conversations,type_context):
    """
    过滤对话中的重复图片
    
    规则：
    1. 在同一个 conversation 中，相同的图片路径只保留第一次出现
    2. 后续出现的相同图片会被删除
    3. 如果整个 content 只有图片被删除，则保留其他内容
    
    Args:
        conversations: 嵌套的对话列表
    
    Returns:
        过滤后的对话列表
    """
    
    # 深拷贝避免修改原数据
    filtered_conversations = copy.deepcopy(conversations)
    
    for conversation in filtered_conversations:
        # 临时变量：记录当前对话中已经出现过的图片路径
        seen_images = set()
        
        for turn in conversation:
            if turn["role"] == "user" and "content" in turn:
                if not isinstance(turn["content"], list):
                    continue
                # 记录要删除的 content 索引
                contents_to_remove = []
                
                for idx, content in enumerate(turn["content"]):
                    print(f"\ncontent:{content}")
                    if content.get("type") == type_context and type_context in content:
                        image_path = content[type_context]
                        
                        # 如果这个图片已经出现过，标记删除
                        if image_path in seen_images:
                            contents_to_remove.append(idx)
                        else:
                            # 第一次出现，记录下来
                            seen_images.add(image_path)
                
                # 倒序删除（避免索引错乱）
                for idx in reversed(contents_to_remove):
                    turn["content"].pop(idx)
    
    return filtered_conversations

class InferenceRequest:
    """推理请求数据结构"""
    def __init__(self, request_id: str, messages: List[Dict], result_queue: queue.Queue):
        self.request_id = request_id
        self.messages = messages
        self.result_queue = result_queue

class LLava_BatchInferenceManager:
    def __init__(self, model=None, root_path=None, batch_size=5, timeout=0.5):
        # 保存模型和处理器
        self.model = model 
        self.thread_name = f"BatchThread-{model}-{id(self)}"
        start_time = time.time()
        self.LLava_model, self.processor = setup_model(model,root_path)
        end_time = time.time()
        print(f"\n加载模型耗时:{end_time-start_time}")
       
        self.batch_size = batch_size  # 批大小，例如5表示攒够5个请求再推理
        self.timeout = timeout  # 超时时间，防止队列一直等不满
        
        # 创建请求队列（线程安全的FIFO队列）
        self.request_queue = queue.Queue()  
        # 工作线程会从这个队列取请求，主线程通过submit_request放入请求
        
        # 创建停止事件（用于优雅地关闭后台线程）
        self.stop_event = threading.Event()  
        # 当调用stop()时，会设置这个事件，后台线程检测到后退出循环
        
        self.batch_thread = None  # 后台批处理线程的引用（初始为空）
        
    def start(self):
        """启动批量推理线程"""
        # 检查线程是否已存在且在运行
        if self.batch_thread is None or not self.batch_thread.is_alive():
            # self.batch_thread is None: 第一次启动
            # not self.batch_thread.is_alive(): 线程已死亡，需要重启
            
            self.stop_event.clear()  # 清除停止标志（重置为未停止状态）
            
            # 创建后台线程
            self.batch_thread = threading.Thread(
                target=self._batch_inference_loop,  # 线程执行的函数
                daemon=True  # 守护线程：主程序退出时自动结束，不阻塞程序退出
            )
            self.batch_thread.start()  # 启动线程（开始执行_batch_inference_loop）
            print("[INFO] Batch inference thread started")
    
    def stop(self):
        """停止批量推理线程"""
        self.stop_event.set()  # 设置停止事件（通知后台线程退出）
        # set()会让_batch_inference_loop中的self.stop_event.is_set()返回True
        
        if self.batch_thread:  # 如果线程存在
            self.batch_thread.join()  # 等待线程结束（阻塞直到线程退出）
            # join()确保线程完全结束后才继续执行，避免资源泄漏
    
    def submit_request(self, messages: List[Dict]) -> Tuple[str, int, int]:
        """
        提交推理请求（阻塞直到获得结果）
        
        Args:
            messages: 对话历史，格式如[{"role": "user", "content": [...]}, ...]
        
        Returns:
            (output_text, prompt_tokens, completion_tokens)
            输出文本、输入token数、输出token数
        """
        request_id = str(uuid.uuid4())  # 生成唯一的请求ID（用于调试追踪）
        result_queue = queue.Queue()  # 为这个请求创建专属的结果队列
        # 后台线程会将结果放入这个队列，当前线程从这里取结果
        
        # 创建请求对象
        request = InferenceRequest(
            request_id=request_id,  # 请求ID
            messages=messages,  # 要推理的消息
            result_queue=result_queue  # 结果返回通道
        )
        
        self.request_queue.put(request)  # 将请求放入全局队列（非阻塞）
        # 后台线程会从request_queue中取出这个请求
        
        # 阻塞等待结果
        result = result_queue.get()  # 阻塞当前线程，直到后台线程放入结果
        # get()会一直等待，直到result_queue中有数据
        
        if isinstance(result, Exception):  # 如果后台线程返回的是异常
            raise result  # 在当前线程中抛出异常
        
        return result  # 返回推理结果 (output_text, prompt_tokens, completion_tokens)
    
    def _batch_inference_loop(self):
        """批量推理主循环（在后台线程中运行）"""
        while not self.stop_event.is_set():  # 只要没有收到停止信号就持续运行
            # is_set()返回False表示继续运行，返回True表示需要退出
            
            batch_requests = []  # 本次批处理的请求列表（清空重新收集）
            
            # ===== 第一阶段：收集一批请求 =====
            try:
                # 阻塞等待第一个请求（最多等1秒）
                first_request = self.request_queue.get(timeout=1.0)
                # 如果1秒内没有请求到来，会抛出queue.Empty异常
                # timeout避免线程永久阻塞，让它有机会检查stop_event
                
                batch_requests.append(first_request)  # 将第一个请求加入批处理列表
                
                # 尝试收集更多请求（带超时）
                deadline = time.time() + self.timeout  # 计算截止时间
                # 例如：当前时间10:00:00，timeout=0.5，则deadline=10:00:00.5
                
                while len(batch_requests) < self.batch_size:  
                    # 循环直到攒够batch_size个请求（或超时）
                    
                    remaining_time = deadline - time.time()  # 计算剩余等待时间
                    # 例如：当前10:00:00.2，则剩余0.3秒
                    
                    if remaining_time <= 0:  # 如果已经超过截止时间
                        break  # 不再等待，开始处理现有请求
                    
                    try:
                        # 尝试从队列取请求（最多等待剩余时间）
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)  # 成功取到，加入批处理
                    except queue.Empty:  # 超时还没取到新请求
                        break  # 不再等待，开始处理现有请求
                
            except queue.Empty:  # 第一个请求等待超时（1秒内没有任何请求）
                continue  # 回到while开头，重新等待
            
            if not batch_requests:  # 如果请求列表为空（理论上不会发生）
                continue  # 跳过本次循环
            
            # ===== 第二阶段：执行批量推理 =====
            try:
                results = self._batch_generate(batch_requests)  
                # 调用批量推理函数，返回结果列表
                # results格式：[(output1, prompt_tokens1, completion_tokens1), ...]
                
                # 将结果返回给对应的线程
                for request, result in zip(batch_requests, results):
                    # zip配对：第1个请求对应第1个结果，第2个对应第2个...
                    request.result_queue.put(result)  # 将结果放入该请求的专属队列
                    # 这会唤醒submit_request中阻塞等待的线程
                    
            except Exception as e:  # 如果批量推理出错
                # 出错时通知所有请求
                for request in batch_requests:
                    request.result_queue.put(e)  # 将异常放入结果队列
                    # submit_request会检测到异常并重新抛出


    def _batch_generate(self, batch_requests: List[InferenceRequest]) -> List[Tuple[str, int, int]]:
        batch_messages = [deepcopy(req.messages) for req in batch_requests]
        # batch_messages=filter_duplicate_images(batch_messages,"image")
        
        print(f"\n[BATCH] Processing {len(batch_messages)} requests")
        start_time = time.time()
        images = []
        prompts = []
        # videos = []
        print(f"\nbatch_messages:{batch_messages}")
        for conversation in batch_messages:
            # 提取当前对话中的所有图片路径，并同时修改原对话结构
            image = []
            for turn in conversation:
                if turn["role"] == "user":
                    if not isinstance(turn["content"], list):
                        continue
                    for content in turn["content"]:
                        if content["type"] == "image" and "image" in content:
                            image_path = content["image"]
                            image.append(Image.open(image_path))
                            del content["image"]
            if image:
                images.append(image)

            prompts.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True))
        print(f"\ncleaned_batch_prompts:{prompts}")
        print(f"\ncleaned_batch_images:{images}")
        if images:
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.LLava_model.device)
        # 处理视觉信息（图片/视频）
        else:
            inputs = self.processor( text=prompts, padding=True, return_tensors="pt").to(self.LLava_model.device)
        # input_token_count = inputs['input_ids'].shape[1]  # 每个样本的输入 token 数
        # batch_size = inputs['input_ids'].shape[0]  # batch 大小
        generate_ids = self.LLava_model.generate(**inputs, max_new_tokens=300)

        # 统计输出 token 数量
        # output_token_count = generate_ids.shape[1]  # 包含输入+生成的 token
        # generated_token_count = output_token_count - input_token_count  # 仅生成的 token

        text_prompt =  self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        all_results = []
        # 提取每个对话中最后一个 ASSISTANT: 之后的内容
        for idx, text in enumerate(text_prompt):
            # 统计当前样本的实际 token 数（不包括 padding）
            actual_input_tokens = (inputs['input_ids'][idx] !=  self.processor.tokenizer.pad_token_id).sum().item()
            actual_output_tokens = (generate_ids[idx] !=  self.processor.tokenizer.pad_token_id).sum().item()
            actual_generated_tokens = actual_output_tokens - actual_input_tokens
            
            # 按 "ASSISTANT:" 分割
            parts = text.split("ASSISTANT:")
            # 获取最后一个部分并去除首尾空格
            last_response = parts[-1].strip()
            all_results.append((
                last_response,
                actual_input_tokens,
                actual_generated_tokens
            ))
        end_time = time.time()
        print(f"[BATCH] Completed in {end_time - start_time:.2f}s")
        
        return all_results

class Video_LLava_BatchInferenceManager:
    def __init__(self, model=None, root_path=None, batch_size=5, timeout=0.5):
        # 保存模型和处理器
        self.model = model 
        self.thread_name = f"BatchThread-{model}-{id(self)}"
        start_time = time.time()
        self.LLava_model, self.processor = setup_model(model,root_path)
        end_time = time.time()
        print(f"\n加载模型耗时:{end_time-start_time}")
       
        self.batch_size = batch_size  # 批大小，例如5表示攒够5个请求再推理
        self.timeout = timeout  # 超时时间，防止队列一直等不满
        
        # 创建请求队列（线程安全的FIFO队列）
        self.request_queue = queue.Queue()  
        # 工作线程会从这个队列取请求，主线程通过submit_request放入请求
        
        # 创建停止事件（用于优雅地关闭后台线程）
        self.stop_event = threading.Event()  
        # 当调用stop()时，会设置这个事件，后台线程检测到后退出循环
        
        self.batch_thread = None  # 后台批处理线程的引用（初始为空）
        
    def start(self):
        """启动批量推理线程"""
        # 检查线程是否已存在且在运行
        if self.batch_thread is None or not self.batch_thread.is_alive():
            # self.batch_thread is None: 第一次启动
            # not self.batch_thread.is_alive(): 线程已死亡，需要重启
            
            self.stop_event.clear()  # 清除停止标志（重置为未停止状态）
            
            # 创建后台线程
            self.batch_thread = threading.Thread(
                target=self._batch_inference_loop,  # 线程执行的函数
                daemon=True  # 守护线程：主程序退出时自动结束，不阻塞程序退出
            )
            self.batch_thread.start()  # 启动线程（开始执行_batch_inference_loop）
            print("[INFO] Batch inference thread started")
    
    def stop(self):
        """停止批量推理线程"""
        self.stop_event.set()  # 设置停止事件（通知后台线程退出）
        # set()会让_batch_inference_loop中的self.stop_event.is_set()返回True
        
        if self.batch_thread:  # 如果线程存在
            self.batch_thread.join()  # 等待线程结束（阻塞直到线程退出）
            # join()确保线程完全结束后才继续执行，避免资源泄漏
    
    def submit_request(self, messages: List[Dict]) -> Tuple[str, int, int]:
        """
        提交推理请求（阻塞直到获得结果）
        
        Args:
            messages: 对话历史，格式如[{"role": "user", "content": [...]}, ...]
        
        Returns:
            (output_text, prompt_tokens, completion_tokens)
            输出文本、输入token数、输出token数
        """
        request_id = str(uuid.uuid4())  # 生成唯一的请求ID（用于调试追踪）
        result_queue = queue.Queue()  # 为这个请求创建专属的结果队列
        # 后台线程会将结果放入这个队列，当前线程从这里取结果
        
        # 创建请求对象
        request = InferenceRequest(
            request_id=request_id,  # 请求ID
            messages=messages,  # 要推理的消息
            result_queue=result_queue  # 结果返回通道
        )
        
        self.request_queue.put(request)  # 将请求放入全局队列（非阻塞）
        # 后台线程会从request_queue中取出这个请求
        
        # 阻塞等待结果
        result = result_queue.get()  # 阻塞当前线程，直到后台线程放入结果
        # get()会一直等待，直到result_queue中有数据
        
        if isinstance(result, Exception):  # 如果后台线程返回的是异常
            raise result  # 在当前线程中抛出异常
        
        return result  # 返回推理结果 (output_text, prompt_tokens, completion_tokens)
    
    def _batch_inference_loop(self):
        """批量推理主循环（在后台线程中运行）"""
        while not self.stop_event.is_set():  # 只要没有收到停止信号就持续运行
            # is_set()返回False表示继续运行，返回True表示需要退出
            
            batch_requests = []  # 本次批处理的请求列表（清空重新收集）
            
            # ===== 第一阶段：收集一批请求 =====
            try:
                # 阻塞等待第一个请求（最多等1秒）
                first_request = self.request_queue.get(timeout=1.0)
                # 如果1秒内没有请求到来，会抛出queue.Empty异常
                # timeout避免线程永久阻塞，让它有机会检查stop_event
                
                batch_requests.append(first_request)  # 将第一个请求加入批处理列表
                
                # 尝试收集更多请求（带超时）
                deadline = time.time() + self.timeout  # 计算截止时间
                # 例如：当前时间10:00:00，timeout=0.5，则deadline=10:00:00.5
                
                while len(batch_requests) < self.batch_size:  
                    # 循环直到攒够batch_size个请求（或超时）
                    
                    remaining_time = deadline - time.time()  # 计算剩余等待时间
                    # 例如：当前10:00:00.2，则剩余0.3秒
                    
                    if remaining_time <= 0:  # 如果已经超过截止时间
                        break  # 不再等待，开始处理现有请求
                    
                    try:
                        # 尝试从队列取请求（最多等待剩余时间）
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)  # 成功取到，加入批处理
                    except queue.Empty:  # 超时还没取到新请求
                        break  # 不再等待，开始处理现有请求
                
            except queue.Empty:  # 第一个请求等待超时（1秒内没有任何请求）
                continue  # 回到while开头，重新等待
            
            if not batch_requests:  # 如果请求列表为空（理论上不会发生）
                continue  # 跳过本次循环
            
            # ===== 第二阶段：执行批量推理 =====
            try:
                results = self._batch_generate(batch_requests)  
                # 调用批量推理函数，返回结果列表
                # results格式：[(output1, prompt_tokens1, completion_tokens1), ...]
                
                # 将结果返回给对应的线程
                for request, result in zip(batch_requests, results):
                    # zip配对：第1个请求对应第1个结果，第2个对应第2个...
                    request.result_queue.put(result)  # 将结果放入该请求的专属队列
                    # 这会唤醒submit_request中阻塞等待的线程
                    
            except Exception as e:  # 如果批量推理出错
                # 出错时通知所有请求
                for request in batch_requests:
                    request.result_queue.put(e)  # 将异常放入结果队列
                    # submit_request会检测到异常并重新抛出


    def _batch_generate(self, batch_requests: List[InferenceRequest]) -> List[Tuple[str, int, int]]:
        batch_messages = [deepcopy(req.messages) for req in batch_requests]
        # batch_messages=filter_duplicate_images(batch_messages,"video")
        
        print(f"\n[BATCH] Processing {len(batch_messages)} requests")
        start_time = time.time()
        images = []
        prompts = []
        videos = []
        clip=None

        for conversation in batch_messages:
            # 提取当前对话中的所有图片路径，并同时修改原对话结构
            for turn in conversation:
                if turn["role"] == "user":
                    if not isinstance(turn["content"], list):
                        continue
                    for content in turn["content"]:
                        if content["type"] == "video" and "video" in content:
                            video_path = content["video"]
                            container=av.open(video_path)
                            total_frames = container.streams.video[0].frames
                            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                            clip = read_video_pyav(container, indices)
                            videos.append(clip)
                            del content["video"]
            prompts.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True))
        # if videos:
        inputs = self.processor(videos=videos, text=prompts, padding=True, return_tensors="pt").to(self.LLava_model.device)
        generate_ids = self.LLava_model.generate(**inputs, max_new_tokens=1000)
        text_prompt =  self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        all_results = []
        # 提取每个对话中最后一个 ASSISTANT: 之后的内容
        for idx, text in enumerate(text_prompt):
            # 统计当前样本的实际 token 数（不包括 padding）
            actual_input_tokens = (inputs['input_ids'][idx] !=  self.processor.tokenizer.pad_token_id).sum().item()
            actual_output_tokens = (generate_ids[idx] !=  self.processor.tokenizer.pad_token_id).sum().item()
            actual_generated_tokens = actual_output_tokens - actual_input_tokens
            
            # 按 "ASSISTANT:" 分割
            parts = text.split("ASSISTANT:")
            # 获取最后一个部分并去除首尾空格
            last_response = parts[-1].strip()
            all_results.append((
                last_response,
                actual_input_tokens,
                actual_generated_tokens
            ))
        end_time = time.time()
        print(f"[BATCH] Completed in {end_time - start_time:.2f}s")
        
        return all_results


class BatchInferenceManager:
    
    def __init__(self, model=None, root_path=None, batch_size=5, timeout=0.5):

        self.model = model  # 用于执行generate的模型
        # self.model_path=Path(root_path) / 'models' / model
        self.thread_name = f"BatchThread-{model}-{id(self)}"
        start_time = time.time()
        self.Qwen_model, self.processor = setup_model(model,root_path)
        self.processor.tokenizer.padding_side = "left"
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        eos_id = self.processor.tokenizer.eos_token_id

        self.Qwen_model.generation_config.pad_token_id = pad_id
        self.Qwen_model.generation_config.eos_token_id = eos_id
     
        end_time = time.time()
        print(f"\n加载模型耗时:{end_time-start_time}")
       
        self.batch_size = batch_size  # 批大小，例如5表示攒够5个请求再推理
        self.timeout = timeout  # 超时时间，防止队列一直等不满
        
        # 创建请求队列（线程安全的FIFO队列）
        self.request_queue = queue.Queue()  
        # 工作线程会从这个队列取请求，主线程通过submit_request放入请求
        
        # 创建停止事件（用于优雅地关闭后台线程）
        self.stop_event = threading.Event()  
        # 当调用stop()时，会设置这个事件，后台线程检测到后退出循环
        
        self.batch_thread = None  # 后台批处理线程的引用（初始为空）
        
    def start(self):
        """启动批量推理线程"""
        # 检查线程是否已存在且在运行
        if self.batch_thread is None or not self.batch_thread.is_alive():
            # self.batch_thread is None: 第一次启动
            # not self.batch_thread.is_alive(): 线程已死亡，需要重启
            
            self.stop_event.clear()  # 清除停止标志（重置为未停止状态）
            
            # 创建后台线程
            self.batch_thread = threading.Thread(
                target=self._batch_inference_loop,  # 线程执行的函数
                daemon=True  # 守护线程：主程序退出时自动结束，不阻塞程序退出
            )
            self.batch_thread.start()  # 启动线程（开始执行_batch_inference_loop）
            print("[INFO] Batch inference thread started")
    
    def stop(self):
        """停止批量推理线程"""
        self.stop_event.set()  # 设置停止事件（通知后台线程退出）
        # set()会让_batch_inference_loop中的self.stop_event.is_set()返回True
        
        if self.batch_thread:  # 如果线程存在
            self.batch_thread.join()  # 等待线程结束（阻塞直到线程退出）
            # join()确保线程完全结束后才继续执行，避免资源泄漏
    
    def submit_request(self, messages: List[Dict]) -> Tuple[str, int, int]:
        """
        提交推理请求（阻塞直到获得结果）
        
        Args:
            messages: 对话历史，格式如[{"role": "user", "content": [...]}, ...]
        
        Returns:
            (output_text, prompt_tokens, completion_tokens)
            输出文本、输入token数、输出token数
        """
        request_id = str(uuid.uuid4())  # 生成唯一的请求ID（用于调试追踪）
        result_queue = queue.Queue()  # 为这个请求创建专属的结果队列
        # 后台线程会将结果放入这个队列，当前线程从这里取结果
        
        # 创建请求对象
        request = InferenceRequest(
            request_id=request_id,  # 请求ID
            messages=messages,  # 要推理的消息
            result_queue=result_queue  # 结果返回通道
        )
        
        self.request_queue.put(request)  # 将请求放入全局队列（非阻塞）
        # 后台线程会从request_queue中取出这个请求
        
        # 阻塞等待结果
        result = result_queue.get()  # 阻塞当前线程，直到后台线程放入结果
        # get()会一直等待，直到result_queue中有数据
        
        if isinstance(result, Exception):  # 如果后台线程返回的是异常
            raise result  # 在当前线程中抛出异常
        
        return result  # 返回推理结果 (output_text, prompt_tokens, completion_tokens)
    
    def _batch_inference_loop(self):
        """批量推理主循环（在后台线程中运行）"""
        while not self.stop_event.is_set():  # 只要没有收到停止信号就持续运行
            # is_set()返回False表示继续运行，返回True表示需要退出
            
            batch_requests = []  # 本次批处理的请求列表（清空重新收集）
            
            # ===== 第一阶段：收集一批请求 =====
            try:
                # 阻塞等待第一个请求（最多等1秒）
                first_request = self.request_queue.get(timeout=1.0)
                # 如果1秒内没有请求到来，会抛出queue.Empty异常
                # timeout避免线程永久阻塞，让它有机会检查stop_event
                
                batch_requests.append(first_request)  # 将第一个请求加入批处理列表
                
                # 尝试收集更多请求（带超时）
                deadline = time.time() + self.timeout  # 计算截止时间
                # 例如：当前时间10:00:00，timeout=0.5，则deadline=10:00:00.5
                
                while len(batch_requests) < self.batch_size:  
                    # 循环直到攒够batch_size个请求（或超时）
                    
                    remaining_time = deadline - time.time()  # 计算剩余等待时间
                    # 例如：当前10:00:00.2，则剩余0.3秒
                    
                    if remaining_time <= 0:  # 如果已经超过截止时间
                        break  # 不再等待，开始处理现有请求
                    
                    try:
                        # 尝试从队列取请求（最多等待剩余时间）
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)  # 成功取到，加入批处理
                    except queue.Empty:  # 超时还没取到新请求
                        break  # 不再等待，开始处理现有请求
                
            except queue.Empty:  # 第一个请求等待超时（1秒内没有任何请求）
                continue  # 回到while开头，重新等待
            
            if not batch_requests:  # 如果请求列表为空（理论上不会发生）
                continue  # 跳过本次循环
            
            # ===== 第二阶段：执行批量推理 =====
            try:
                results = self._batch_generate(batch_requests)  
                # 调用批量推理函数，返回结果列表
                # results格式：[(output1, prompt_tokens1, completion_tokens1), ...]
                
                # 将结果返回给对应的线程
                for request, result in zip(batch_requests, results):
                    # zip配对：第1个请求对应第1个结果，第2个对应第2个...
                    request.result_queue.put(result)  # 将结果放入该请求的专属队列
                    # 这会唤醒submit_request中阻塞等待的线程
                    
            except Exception as e:  # 如果批量推理出错
                # 出错时通知所有请求
                for request in batch_requests:
                    request.result_queue.put(e)  # 将异常放入结果队列
                    # submit_request会检测到异常并重新抛出


    def _batch_generate(self, batch_requests: List[InferenceRequest]) -> List[Tuple[str, int, int]]:
        """
        批量执行推理
        
        Returns:
            [(output_text, prompt_tokens, completion_tokens), ...]
        """
        batch_messages = [req.messages for req in batch_requests]

        # batch_messages=filter_duplicate_images(batch_messages,"image")
        
        # print(f"\n[BATCH] Processing {len(batch_messages)} requests")
        start_time = time.time()
        texts = []
        for messages in batch_messages:
        # 使用 processor 格式化消息
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        # 处理视觉信息（图片/视频）
        image_inputs, video_inputs = process_vision_info(batch_messages)
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        # Tokenize
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.Qwen_model.device)
        # 统计实际输入 token 数量（排除 padding）
        actual_input_tokens = []
        for i in range(len(batch_messages)):
            input_ids = inputs.input_ids[i]
            non_pad_tokens = (input_ids != pad_id).sum().item()
            actual_input_tokens.append(non_pad_tokens)

        # 生成
        with torch.no_grad():
            generated_ids = self.Qwen_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )

        # 移除输入部分，只保留生成的内容
        generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解码
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
       
        end_time = time.time()
        print(f"\n生成时间: {end_time - start_time:.2f}s")
        all_results = []
        for i in range(len(output_texts)):
            input_length = inputs.input_ids[i].shape[0]
            output_length = generated_ids[i].shape[0]
            completion_tokens = output_length - input_length
            
            all_results.append((
                output_texts[i],
                actual_input_tokens[i],  # 使用实际的输入 token 数量
                completion_tokens
            ))
    
        end_time = time.time()
        print(f"[BATCH] Completed in {end_time - start_time:.2f}s")
        
        return all_results

class Video_BatchInferenceManager:
    def __init__(self, model=None, root_path=None, batch_size=5, timeout=0.5):
        self.model = model  # 用于执行generate的模型
        # self.model_path=Path(root_path) / 'models' / model
        start_time = time.time()
        self.Qwen_model, self.processor = setup_model(model,root_path)
        self.processor.tokenizer.padding_side = "left"
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        eos_id = self.processor.tokenizer.eos_token_id

        self.Qwen_model.generation_config.pad_token_id = pad_id
        self.Qwen_model.generation_config.eos_token_id = eos_id
    
        end_time = time.time()
        print(f"\n加载模型耗时:{end_time-start_time}")
       
        self.batch_size = batch_size  # 批大小，例如5表示攒够5个请求再推理
        self.timeout = timeout  # 超时时间，防止队列一直等不满
        
        # 创建请求队列（线程安全的FIFO队列）
        self.request_queue = queue.Queue()  
        # 工作线程会从这个队列取请求，主线程通过submit_request放入请求
        
        # 创建停止事件（用于优雅地关闭后台线程）
        self.stop_event = threading.Event()  
        # 当调用stop()时，会设置这个事件，后台线程检测到后退出循环
        
        self.batch_thread = None  # 后台批处理线程的引用（初始为空）
        
    def start(self):
        """启动批量推理线程"""
        # 检查线程是否已存在且在运行
        if self.batch_thread is None or not self.batch_thread.is_alive():
            # self.batch_thread is None: 第一次启动
            # not self.batch_thread.is_alive(): 线程已死亡，需要重启
            
            self.stop_event.clear()  # 清除停止标志（重置为未停止状态）
            
            # 创建后台线程
            self.batch_thread = threading.Thread(
                target=self._batch_inference_loop,  # 线程执行的函数
                daemon=True  # 守护线程：主程序退出时自动结束，不阻塞程序退出
            )
            self.batch_thread.start()  # 启动线程（开始执行_batch_inference_loop）
            print("[INFO] Batch inference thread started")
    
    def stop(self):
        """停止批量推理线程"""
        self.stop_event.set()  # 设置停止事件（通知后台线程退出）
        # set()会让_batch_inference_loop中的self.stop_event.is_set()返回True
        
        if self.batch_thread:  # 如果线程存在
            self.batch_thread.join()  # 等待线程结束（阻塞直到线程退出）
            # join()确保线程完全结束后才继续执行，避免资源泄漏
    
    def submit_request(self, messages: List[Dict]) -> Tuple[str, int, int]:
        request_id = str(uuid.uuid4())  # 生成唯一的请求ID（用于调试追踪）
        result_queue = queue.Queue()  # 为这个请求创建专属的结果队列
        # 后台线程会将结果放入这个队列，当前线程从这里取结果
        
        # 创建请求对象
        request = InferenceRequest(
            request_id=request_id,  # 请求ID
            messages=messages,  # 要推理的消息
            result_queue=result_queue  # 结果返回通道
        )
        
        self.request_queue.put(request)  # 将请求放入全局队列（非阻塞）
        # 后台线程会从request_queue中取出这个请求
        
        # 阻塞等待结果
        result = result_queue.get()  # 阻塞当前线程，直到后台线程放入结果
        # get()会一直等待，直到result_queue中有数据
        
        if isinstance(result, Exception):  # 如果后台线程返回的是异常
            raise result  # 在当前线程中抛出异常
        
        return result  # 返回推理结果 (output_text, prompt_tokens, completion_tokens)
    
    def _batch_inference_loop(self):
        """批量推理主循环（在后台线程中运行）"""
        while not self.stop_event.is_set():  # 只要没有收到停止信号就持续运行
            # is_set()返回False表示继续运行，返回True表示需要退出
            
            batch_requests = []  # 本次批处理的请求列表（清空重新收集）
            
            # ===== 第一阶段：收集一批请求 =====
            try:
                # 阻塞等待第一个请求（最多等1秒）
                first_request = self.request_queue.get(timeout=1.0)
                # 如果1秒内没有请求到来，会抛出queue.Empty异常
                # timeout避免线程永久阻塞，让它有机会检查stop_event
                
                batch_requests.append(first_request)  # 将第一个请求加入批处理列表
                
                # 尝试收集更多请求（带超时）
                deadline = time.time() + self.timeout  # 计算截止时间
                # 例如：当前时间10:00:00，timeout=0.5，则deadline=10:00:00.5
                
                while len(batch_requests) < self.batch_size:  
                    # 循环直到攒够batch_size个请求（或超时）
                    
                    remaining_time = deadline - time.time()  # 计算剩余等待时间
                    # 例如：当前10:00:00.2，则剩余0.3秒
                    
                    if remaining_time <= 0:  # 如果已经超过截止时间
                        break  # 不再等待，开始处理现有请求
                    
                    try:
                        # 尝试从队列取请求（最多等待剩余时间）
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)  # 成功取到，加入批处理
                    except queue.Empty:  # 超时还没取到新请求
                        break  # 不再等待，开始处理现有请求
                
            except queue.Empty:  # 第一个请求等待超时（1秒内没有任何请求）
                continue  # 回到while开头，重新等待
            
            if not batch_requests:  # 如果请求列表为空（理论上不会发生）
                continue  # 跳过本次循环
            
            # ===== 第二阶段：执行批量推理 =====
            try:
                results = self._batch_generate(batch_requests)  
                # 调用批量推理函数，返回结果列表
                # results格式：[(output1, prompt_tokens1, completion_tokens1), ...]
                
                # 将结果返回给对应的线程
                for request, result in zip(batch_requests, results):
                    # zip配对：第1个请求对应第1个结果，第2个对应第2个...
                    request.result_queue.put(result)  # 将结果放入该请求的专属队列
                    # 这会唤醒submit_request中阻塞等待的线程
                    
            except Exception as e:  # 如果批量推理出错
                # 出错时通知所有请求
                for request in batch_requests:
                    request.result_queue.put(e)  # 将异常放入结果队列
                    # submit_request会检测到异常并重新抛出


    def _batch_generate(self, batch_requests: List[InferenceRequest]) -> List[Tuple[str, int, int]]:
        """
        批量执行推理
        
        Returns:
            [(output_text, prompt_tokens, completion_tokens), ...]
        """
        batch_messages = [req.messages for req in batch_requests]
        # batch_messages=filter_duplicate_images(batch_messages,"video")
        print(f"\nbatch_messages:{batch_messages}")
        # print(f"\n[BATCH] Processing {len(batch_messages)} requests")
        start_time = time.time()
        texts = []
        for messages in batch_messages:
        # 使用 processor 格式化消息
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)

        # 处理视觉信息（图片/视频）
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        # Tokenize
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.Qwen_model.device)

        actual_input_tokens = []
        for i in range(len(batch_messages)):
            input_ids = inputs.input_ids[i]
            non_pad_tokens = (input_ids != pad_id).sum().item()
            actual_input_tokens.append(non_pad_tokens)

        # 生成
        with torch.no_grad():
            generated_ids = self.Qwen_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )

        # 移除输入部分，只保留生成的内容
        generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解码
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
       
        end_time = time.time()
        print(f"\n生成时间: {end_time - start_time:.2f}s")
        all_results = []
        for i in range(len(output_texts)):
            input_length = inputs.input_ids[i].shape[0]
            output_length = generated_ids[i].shape[0]
            completion_tokens = output_length - input_length
            
            all_results.append((
                output_texts[i],
                actual_input_tokens[i],  # 使用实际的输入 token 数量
                completion_tokens
            ))
      
    
        end_time = time.time()
        print(f"[BATCH] Completed in {end_time - start_time:.2f}s")
        
        return all_results

def qwen_generate_content(messages,batch_manager):
    return batch_manager.submit_request(messages)
    