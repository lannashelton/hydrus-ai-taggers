print("DEBUG: Loading interrogate.py...")

import os
import re

# Set environment variables for libraries that might ignore ONNX settings
threads_env = os.environ.get("OMP_NUM_THREADS")
if threads_env:
    try:
        t_count = int(threads_env)
        os.environ["OPENBLAS_NUM_THREADS"] = str(t_count)
        os.environ["MKL_NUM_THREADS"] = str(t_count)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(t_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(t_count)
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    except:
        pass

import pandas as pd
import numpy as np
import cv2
from typing import Tuple, Dict, List
from PIL import Image
from pathlib import Path

import dbimutils

# Apply OpenCV limits immediately
if threads_env:
    cv2.setNumThreads(int(threads_env))
    cv2.ocl.setUseOpenCL(False)

tag_escape_pattern = re.compile(r'([\\()])')

class WaifuDiffusionInterrogator:
    def __init__(
            self,
            name: str,
            model_file: str,
            tags_file: str,
            folder: str,
            ratingsflag: bool,
            numberofratings: int,
            **kwargs
    ) -> None:
        self.name = name
        self.model_file = model_file
        self.tags_file = tags_file
        self.folder = folder
        self.ratingsflag = ratingsflag
        self.numberofratings = numberofratings
        self.kwargs = kwargs
        self.model = None
        self.device = "CPU"          # will be set by load()

    def findpaths(self) -> Tuple[Path, Path]:
        local_base = Path('model') / self.folder
        if (local_base / self.model_file).exists():
            return local_base / self.model_file, local_base / self.tags_file

        parent_base = Path('../model') / self.folder
        if (parent_base / self.model_file).exists():
            return parent_base / self.model_file, parent_base / self.tags_file

        return local_base / self.model_file, local_base / self.tags_file

    def load(self, device: str = "CPU") -> None:
        """
        Load the ONNX model.
        device : "CPU", "GPU" (CUDA), or "NPU" (Intel NPU via OpenVINO EP)
        """
        self.device = device.upper()
        model_file, tags_file = self.findpaths()

        from onnxruntime import InferenceSession, SessionOptions, ExecutionMode

        opts = SessionOptions()
        t_env = os.environ.get("OMP_NUM_THREADS")
        if t_env:
            threads = int(t_env)
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = threads
            opts.execution_mode = ExecutionMode.ORT_SEQUENTIAL

        device_upper = self.device
        if device_upper == "GPU":
            providers = ['CUDAExecutionProvider']
            provider_options = [{
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'DEFAULT'
            }]
        elif device_upper == "NPU":
            # OpenVINO EP with NPU precision
            providers = [
                ('OpenVINOExecutionProvider', {
                    'device_type': 'NPU',
                    'precision': 'FP16'
                }),
                'CPUExecutionProvider'
            ]
            provider_options = None   # not used when providers is list of tuples
        else:   # CPU
            providers = ['CPUExecutionProvider']
            provider_options = [{}]

        try:
            print(f"DEBUG: Loading model with provider: {providers[0]}")
            self.model = InferenceSession(
                str(model_file),
                providers=providers,
                provider_options=provider_options,
                sess_options=opts
            )
            session_provider = self.model.get_providers()[0]
            print(f"DEBUG: Success! Loaded with provider: {session_provider}")
        except Exception as e:
            print(f"DEBUG: {device_upper} initialization failed ({str(e)[:80]}). Falling back to CPU.")
            providers = ['CPUExecutionProvider']
            provider_options = [{}]
            self.model = InferenceSession(
                str(model_file),
                providers=providers,
                provider_options=provider_options,
                sess_options=opts
            )
            print("DEBUG: Successfully loaded with CPU fallback")

        print(f'Loaded {self.name} model from {model_file}')
        self.tags = pd.read_csv(tags_file)

    def _prepare_image(self, image: Image, height: int) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGBA')
            new_image = Image.new('RGBA', image.size, 'WHITE')
            new_image.paste(image, mask=image)
            image = new_image.convert('RGB')

        img_np = np.asarray(image)
        img_np = img_np[:, :, ::-1]   # RGB to BGR
        img_np = dbimutils.make_square(img_np, height)
        img_np = dbimutils.smart_resize(img_np, height)
        img_np = img_np.astype(np.float32)
        return img_np

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        ratings_list, tags_list = self.interrogate_batch([image])
        return ratings_list[0], tags_list[0]

    def interrogate_batch(self, images: List[Image]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        _, height, _, _ = self.model.get_inputs()[0].shape

        # 1. Preprocess
        batch_input = []
        for img in images:
            processed = self._prepare_image(img, height)
            batch_input.append(processed)

        # 2. Stack
        input_tensor = np.stack(batch_input, axis=0)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name

        # 3. Inference
        confidents_batch = self.model.run([label_name], {input_name: input_tensor})[0]

        # 4. Process Output
        batch_ratings = []
        batch_tags = []

        tag_names = self.tags['name'].values

        for i in range(len(images)):
            confidents = confidents_batch[i]

            if confidents.ndim > 1:
                confidents = confidents.flatten()

            result_dict = dict(zip(tag_names, confidents))

            if self.ratingsflag:
                ratings = {k: result_dict[k] for k in list(result_dict)[:self.numberofratings]}
                tags = {k: result_dict[k] for k in list(result_dict)[self.numberofratings:]}
            else:
                ratings = {}
                tags = result_dict

            batch_ratings.append(ratings)
            batch_tags.append(tags)

        return batch_ratings, batch_tags
