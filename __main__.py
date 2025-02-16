import os
import logging
import concurrent.futures
from typing import List, Tuple

import numpy as np
from PIL import Image
import cv2
from scipy.io.wavfile import write
from moviepy.editor import VideoFileClip, AudioFileClip
import tqdm

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 可配置参数
class Config:
    FPS = 30          # 提高帧率获得更流畅动画
    IMG_SIZE = 128    # 增大图像尺寸
    MODE = 1          # 切换彩色模式(一为彩色模式，二为黑白模式)
    DEFAULT_SAMPLE_RATE = 48000  # 专业音频采样率

    @property
    def bytes_per_image(self) -> int:
        if self.MODE == 1:
            return self.IMG_SIZE**2 * 3
        return self.IMG_SIZE**2 // 8

    @property
    def video_codec(self) -> str:
        return 'mp4v' if cv2.__version__ < '4.6' else 'avc1'

config = Config()

# 图像处理核心逻辑
class ImageProcessor:
    @staticmethod
    def process_chunk(chunk: bytes, mode: int, img_size: int) -> Image.Image:
        """处理二进制块生成图像"""
        try:
            if mode == 1:
                pixels = [(chunk[j], chunk[j+1], chunk[j+2]) 
                         for j in range(0, len(chunk), 3)]
                return Image.new('RGB', (img_size, img_size)).putdata(pixels)
            else:
                bits = ''.join(f"{byte:08b}" for byte in chunk)
                pixels = [255 if bit == '1' else 0 for bit in bits]
                return Image.new('L', (img_size, img_size)).putdata(pixels)
        except Exception as e:
            logger.error(f"图像生成失败: {str(e)}")
            raise

# 音频处理优化
class AudioProcessor:
    @staticmethod
    def normalize_audio(data: np.ndarray) -> np.ndarray:
        """音频归一化处理"""
        max_val = np.max(np.abs(data))
        return data / max_val if max_val > 0 else data

    @staticmethod
    def merge_audio_chunks(chunk_files: List[str], output_path: str) -> None:
        """合并音频分块"""
        try:
            audio_clips = [AudioFileClip(f) for f in chunk_files]
            final_clip = concatenate_audioclips(audio_clips)
            final_clip.write_audiofile(output_path)
            for f in chunk_files:
                os.remove(f)
        except Exception as e:
            logger.error(f"音频合并失败: {str(e)}")
            raise

# 视频生成优化
class VideoGenerator:
    @staticmethod
    def images_to_video(images: List[Image.Image], video_path: str) -> None:
        """高效生成视频"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
            with cv2.VideoWriter(
                video_path, 
                fourcc, 
                config.FPS, 
                (config.IMG_SIZE, config.IMG_SIZE), 
                isColor=config.MODE == 1
            ) as video:
                for img in tqdm.tqdm(images, desc="生成视频"):
                    video.write(np.array(img))
        except Exception as e:
            logger.error(f"视频生成失败: {str(e)}")
            raise

# 文件处理优化
class FileProcessor:
    @staticmethod
    def process_binary_chunks(file_path: str, chunk_size: int) -> bytes:
        """流式读取大文件"""
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk

    @staticmethod
    def safe_remove(*paths: str) -> None:
        """安全删除中间文件"""
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"删除文件失败 {path}: {str(e)}")

# 主流程优化
class MediaConverter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def convert_file(self, file_path: str) -> str:
        """完整转换流程"""
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            video_path = os.path.join(self.output_dir, f"{base_name}_video.mp4")
            audio_path = os.path.join(self.output_dir, f"{base_name}_audio.wav")
            final_path = os.path.join(self.output_dir, f"{base_name}_final.mp4")

            # 并行处理图像和音频
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                img_future = executor.submit(self._generate_images, file_path)
                audio_future = executor.submit(self._generate_audio, file_path, audio_path)
                
                images = img_future.result()
                audio_chunks = audio_future.result()

            # 合并音频分块
            if audio_chunks:
                AudioProcessor.merge_audio_chunks(audio_chunks, audio_path)

            # 生成视频并合并
            VideoGenerator.images_to_video(images, video_path)
            self._combine_media(video_path, audio_path, final_path)
            
            return final_path
        except Exception as e:
            logger.error(f"文件转换失败 {file_path}: {str(e)}")
            FileProcessor.safe_remove(video_path, audio_path, final_path)
            raise

    def _generate_images(self, file_path: str) -> List[Image.Image]:
        """生成图像序列"""
        images = []
        bytes_needed = config.bytes_per_image
        for chunk in FileProcessor.process_binary_chunks(file_path, config.CHUNK_SIZE):
            for i in range(0, len(chunk), bytes_needed):
                data_chunk = chunk[i:i+bytes_needed]
                if len(data_chunk) < bytes_needed:
                    data_chunk += b'\x00' * (bytes_needed - len(data_chunk))
                images.append(ImageProcessor.process_chunk(
                    data_chunk, config.MODE, config.IMG_SIZE))
        return images

    def _generate_audio(self, file_path: str, audio_path: str) -> List[str]:
        """生成音频文件"""
        audio_chunks = []
        for i, chunk in enumerate(FileProcessor.process_binary_chunks(file_path, config.CHUNK_SIZE)):
            chunk_path = f"{audio_path}_chunk_{i+1}.wav"
            data = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32) / 255
            data = AudioProcessor.normalize_audio(data)
            write(chunk_path, config.DEFAULT_SAMPLE_RATE, data)
            audio_chunks.append(chunk_path)
        return audio_chunks

    def _combine_media(self, video_path: str, audio_path: str, output_path: str) -> None:
        """合并音视频"""
        try:
            video = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            video = video.set_audio(audio)
            video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                ffmpeg_params=['-crf', '23']  # 控制视频质量
            )
        finally:
            FileProcessor.safe_remove(video_path, audio_path)

# 命令行接口优化
def main():
    input_path = input("请输入文件或文件夹路径：").strip()
    output_folder = input("请输入输出文件夹路径：").strip()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    converter = MediaConverter(output_folder)
    
    if os.path.isfile(input_path):
        try:
            final_path = converter.convert_file(input_path)
            logger.info(f"转换完成: {final_path}")
        except Exception as e:
            logger.error(f"转换失败: {str(e)}")
    elif os.path.isdir(input_path):
        with concurrent.futures.ProcessPoolExecutor(config.MAX_WORKERS) as executor:
            futures = []
            for root, _, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(converter.convert_file, file_path))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"文件处理失败: {str(e)}")
    else:
        logger.error("无效的输入路径")

if __name__ == "__main__":
    main()