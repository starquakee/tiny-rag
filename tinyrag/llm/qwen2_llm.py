import os
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from loguru import logger

from tinyrag.llm.base_llm import BaseLLM

class Qwen2LLM(BaseLLM):
    def __init__(self, model_id_key: str, device: str = "cpu", is_api=False) -> None:
        super().__init__(model_id_key, device, is_api)

        # 从预训练模型加载因果语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_key,  # 模型标识符
            dtype="auto",  # 自动选择张量类型（替代已弃用的 torch_dtype）
            device_map=self.device,  # 分布到特定设备上
            trust_remote_code=True  # 允许加载远程代码
        )
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_key,  # 分词器标识符
            use_fast=False,
            trust_remote_code=True
        )
        # 若无 pad_token，令其与 eos 对齐，避免推理时的 pad/eos 冲突告警
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载配置文件
        self.config = AutoConfig.from_pretrained(
            self.model_id_key,  # 配置文件标识符
            trust_remote_code=True  # 允许加载远程代码
        )

        if self.device == "cpu":
            self.model.float()
        
        # 设置模型为评估模式
        self.model.eval()

    def generate(self, content: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 编码为张量，并显式提供 attention_mask，避免 pad/eos 警告
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.tokenizer, "sep_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id

        # 控制生成长度，启用 cache，显著加速生成，避免“卡住”的观感
        max_ctx = getattr(self.model.config, "max_position_embeddings", 1024)
        max_new = getattr(self, "max_new_tokens", 128)

        logger.info("Qwen2 generate start: max_new_tokens={}, input_len={}", max_new, input_ids.shape[-1])
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                use_cache=True,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        logger.info("Qwen2 generate done. total_len={}", generated_ids.shape[-1])

        # 解码仅新生成的 token
        new_tokens = generated_ids[0, input_ids.shape[-1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response



