import base64
import os
from pathlib import Path

_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}


def encode_image(image_path: str) -> str:
    """
    Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the image file
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    try:
        with open(image_path, "rb") as image_file:

            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        raise IOError(f"Error reading image file: {e}")


def is_video_file(file_path: str) -> bool:
    """检查文件是否是视频文件。"""
    ext = Path(str(file_path)).suffix.lower().lstrip('.')
    return ext in _VIDEO_EXTENSIONS


def encode_media_to_content_parts(file_path: str) -> list:
    """将图片或视频文件编码为 OpenAI 多模态内容列表。
    
    图片返回 1 个 image_url 条目；视频抽帧后返回多个 image_url 条目。
    """
    if is_video_file(file_path):
        from methods.vllm_thread import extract_frames
        frames = extract_frames(str(file_path))
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames]
    else:
        b64 = encode_image(file_path)
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]