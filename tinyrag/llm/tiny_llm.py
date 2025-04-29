import os
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from tinyrag.llm.base_llm import BaseLLM

class TinyLLM(BaseLLM):
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
        sys_text = "你是由wdndev开发的个人助手。"
        # 将模板拆分为“特殊标记 + 普通内容”的片段，特殊标记用单个 token id，内容用正常分词
        segments = [
            ("special", "<|system|>"),
            ("text", sys_text.strip()),
            ("special", "<|user|>"),
            ("text", content.strip()),
            ("special", "<|assistant|>")
        ]
        token_ids = []
        for kind, seg in segments:
            if kind == "special":
                tid = self.tokenizer.convert_tokens_to_ids(seg)
                if tid is None or tid == self.tokenizer.unk_token_id:
                    # 回退为逐字分词（防止未注册为特殊符号时丢失信息）
                    token_ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seg)))
                else:
                    token_ids.append(tid)
            else:
                token_ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seg)))
        # 可选地添加 BOS
        if getattr(self.tokenizer, "bos_token_id", None) is not None:
            token_ids = [self.tokenizer.bos_token_id] + token_ids
        # 截断以适配模型上下文长度，预留生成空间
        max_ctx = getattr(self.model.config, "max_position_embeddings", 1024)
        max_new = 200
        reserve = 8
        keep_len = max(1, max_ctx - max_new - reserve)
        if len(token_ids) > keep_len:
            # 保留 BOS 再拼接尾部
            bos = []
            if getattr(self.tokenizer, "bos_token_id", None) is not None and len(token_ids) > 0 and token_ids[0] == self.tokenizer.bos_token_id:
                bos = [token_ids[0]]
                trunk = token_ids[-keep_len+1:]
                token_ids = bos + trunk
            else:
                token_ids = token_ids[-keep_len:]

        input_ids = torch.tensor([token_ids], device=self.model.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.tokenizer, "sep_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id

        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new,
            use_cache=False,  # 关闭缓存以规避自定义模型对 DynamicCache.seen_tokens 的不兼容
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        # 只保留新生成的 token
        new_tokens = generated_ids[0][len(token_ids):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response
