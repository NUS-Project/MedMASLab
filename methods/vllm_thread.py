from openai import OpenAI
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
from methods.utils import encode_image
from methods.thread import InferenceRequest
import av
import base64
import numpy as np
import copy
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import base64 
import os  
import cv2 


def extract_frames(video_path: str, interval: int = 1, min_frames: int = 4, max_frames: int = 8) -> List[str]:  
    """  
    视频抽帧（自适应调整采样间隔以保证最小帧数）
    :param video_path: 视频文件路径  
    :param interval: 初始采样率,即每N秒抽取一帧  
    :param min_frames: 最小帧数（如果低于此值会自动调整interval）
    :param max_frames: 最大帧数限制
    :return: 抽取的视频帧列表（base64编码）
    """  
    extracted_frames = []  
    
    if not os.path.exists(video_path):
        print(f"  ⚠️ 文件不存在: {video_path}")
        return extracted_frames
    
    video_capture = None
    try:
        video_capture = cv2.VideoCapture(video_path)  
        
        if not video_capture.isOpened():
            print(f"  ⚠️ 无法打开视频")
            return extracted_frames
        
        total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)  
        # width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 检查视频参数有效性
        if frame_rate <= 0 or total_frame_count <= 0:
            print(f"  ⚠️ 视频参数异常: fps={frame_rate}, frames={total_frame_count}")
            return extracted_frames
        
        duration = total_frame_count / frame_rate  # 视频时长（秒）
        
        # 🔥 自适应调整 interval
        frames_interval = int(frame_rate * interval)
        estimated_frames = total_frame_count // frames_interval
        
        # print(f"  → 视频信息: {width}x{height}, {frame_rate:.1f}fps, {total_frame_count}帧, 时长{duration:.1f}秒")
        # print(f"  → 初始设置: interval={interval}秒, 预计抽取{estimated_frames}帧")
        
        # 如果预计帧数低于最小值，自动调整
        if estimated_frames < min_frames:
            # 重新计算 interval 以满足最小帧数要求
            new_interval = duration / min_frames  # 新的秒间隔
            frames_interval = max(1, int(frame_rate * new_interval))  # 转换为帧间隔
            estimated_frames = total_frame_count // frames_interval
            
            print(f"  ⚙️ 自动调整: interval={new_interval:.2f}秒, 预计抽取{estimated_frames}帧")
        
        # 如果预计帧数超过最大值，也进行调整
        if estimated_frames > max_frames:
            new_interval = duration / max_frames
            frames_interval = max(1, int(frame_rate * new_interval))
            estimated_frames = total_frame_count // frames_interval
            
            print(f"  ⚙️ 自动调整: interval={new_interval:.2f}秒, 预计抽取{estimated_frames}帧")
        
        # 开始抽帧
        current_frame = 0  
        frame_count = 0
      
        while current_frame < total_frame_count - 1 and frame_count < max_frames:  
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  
            success, frame = video_capture.read()  
            
            if not success:  
                break  
            
            # 编码为JPEG（不调整尺寸）
            _, buffer = cv2.imencode(".jpg", frame)  
            extracted_frames.append(base64.b64encode(buffer).decode("utf-8"))  
            frame_count += 1
            
            current_frame += frames_interval  
      
        print(f"  ✓ 视频抽帧完成！实际抽取了 {len(extracted_frames)} 帧")
        
    except Exception as e:
        print(f"  ✗ 抽帧失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        if video_capture is not None:
            video_capture.release()
    
    return extracted_frames

def convert_image_to_base64_format(messages: List[dict]) -> List[dict]:
    """
    将消息中的图像格式从本地路径转换为 base64 格式
    输入: {"type": "image", "image": "/path/to/image.jpg"}
    输出: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    def get_image_mime_type(image_path: str) -> str:
        """根据文件扩展名获取 MIME 类型"""
        ext = image_path.lower().split('.')[-1]
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')
    
    # 深拷贝避免修改原始数据
    converted_messages = deepcopy(messages)
    
    for turn in converted_messages:
        if turn["role"] == "user":
            if not isinstance(turn["content"], list):
                continue
            new_content = []  # 用新列表替换
        
            for content in turn["content"]:
                # 处理图片
                if content.get("type") == "image" and "image" in content:
                    image_path = content["image"]
                    base64_image = encode_image(image_path)
                    mime_type = get_image_mime_type(image_path)
                    
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                
                # 处理视频 - 转换为多个图片帧
                elif content.get("type") == "video" and "video" in content:
                    video_path = content["video"]
                    print(f"  → 处理视频: {video_path}")
                    
                    extracted_frames = extract_frames(video_path, interval=3)
                    
                    if not extracted_frames:
                        print(f"  ⚠️ 视频抽帧失败，跳过该视频")
                        continue
                    
                    # 将每一帧作为独立的 image_url 添加
                    for frame_base64 in extracted_frames:
                        new_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_base64}"
                            }
                        })
                
                # 其他类型直接保留
                else:
                    new_content.append(content)
            
            # 替换原 content
            turn["content"] = new_content
                        
                    
    return converted_messages



class VLLMBatchInferenceManager:
    """使用 vLLM API 的批量推理管理器"""
    
    def __init__(self, model=None, root_path=None, batch_size=10, timeout=0.5, vllm_url="http://localhost:8000/v1",api_key="EMPTY"):
        self.model = "gpt-4o-mini"# model
        self.thread_name = f"BatchThread-{model}-{id(self)}"
        self.vllm_url = vllm_url
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_key=api_key
        
        # 初始化 OpenAI 客户端(连接 vLLM)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.vllm_url
        )
        
        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.batch_thread = None
        
        print(f"[INFO] VLLMBatchInferenceManager initialized with model={model}, url={vllm_url}")
        
    def start(self):
        """启动批量推理线程"""
        if self.batch_thread is None or not self.batch_thread.is_alive():
            self.stop_event.clear()
            self.batch_thread = threading.Thread(
                target=self._batch_inference_loop,
                daemon=True
            )
            self.batch_thread.start()
            print("[INFO] VLLM Batch inference thread started")
    
    def stop(self):
        """停止批量推理线程"""
        self.stop_event.set()
        if self.batch_thread:
            self.batch_thread.join()
            print("[INFO] VLLM Batch inference thread stopped")
    
    def submit_request(self, messages: List[Dict]) -> Tuple[str, int, int]:
        """提交推理请求(阻塞直到获得结果)"""
        request_id = str(uuid.uuid4())
        result_queue = queue.Queue()
        
        request = InferenceRequest(
            request_id=request_id,
            messages=messages,
            result_queue=result_queue
        )
        
        self.request_queue.put(request)
        result = result_queue.get()
        
        if isinstance(result, Exception):
            raise result
        
        return result
    
    def _batch_inference_loop(self):
        """批量推理主循环"""
        while not self.stop_event.is_set():
            batch_requests = []
            
            try:
                first_request = self.request_queue.get(timeout=1.0)
                batch_requests.append(first_request)
                
                deadline = time.time() + self.timeout
                while len(batch_requests) < self.batch_size:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        request = self.request_queue.get(timeout=remaining_time)
                        batch_requests.append(request)
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
            
            if not batch_requests:
                continue
            
            try:
                results = self._batch_generate_vllm(batch_requests)
                for request, result in zip(batch_requests, results):
                    request.result_queue.put(result)
            except Exception as e:
                for request in batch_requests:
                    request.result_queue.put(e)
    
    def _batch_generate_vllm(self, batch_requests: List[InferenceRequest]) -> List[Tuple[str, int, int]]:
        """使用 vLLM API 进行批量推理"""
        print(f"\n[VLLM BATCH] Processing {len(batch_requests)} requests")
        start_time = time.time()
        
        all_results = []
        
        # 并发调用 vLLM API(使用线程池加速)
        
        
        def call_vllm_api(messages):
            """调用单个 vLLM API 请求"""
            try:
                messages=convert_image_to_base64_format(messages)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=2048
                )
                
                output_text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                return (output_text, prompt_tokens, completion_tokens)
            except Exception as e:
                print(f"[ERROR] vLLM API call failed: {e}")
                return (f"Error: {str(e)}", 0, 0)
        
        # 使用线程池并发调用(vLLM 内部会自动批处理)
        with ThreadPoolExecutor(max_workers=len(batch_requests)) as executor:
            futures = [executor.submit(call_vllm_api, req.messages) for req in batch_requests]
            all_results = [f.result() for f in futures]
        
        end_time = time.time()
        print(f"[VLLM BATCH] Completed {len(all_results)} requests in {end_time - start_time:.2f}s")
        
        return all_results