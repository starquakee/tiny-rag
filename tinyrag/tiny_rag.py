
import os
import json
import random
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from tinyrag import BaseLLM, Qwen2LLM, TinyLLM
from tinyrag import Searcher
from tinyrag import SentenceSplitter
from tinyrag.utils import write_list_to_jsonl


RAG_PROMPT_TEMPALTE="""参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要简洁，忠于原文，但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""

@dataclass
class RAGConfig:
    base_dir:str = "data/wiki_db"
    llm_model_id:str = "models/tiny_llm_sft_92m"
    emb_model_id: str = "models/bge-small-zh-v1.5"
    ranker_model_id:str = "models/bge-reranker-base"
    device:str = "cpu"
    sent_split_model_id:str = "models/nlp_bert_document-segmentation_chinese-base"
    sent_split_use_model:bool = False
    sentence_size:int = 256
    model_type: str = "tinyllm"
    max_new_tokens: int = 128

def process_docs_text(docs_text, sent_split_model):
    sent_res = sent_split_model.split_text(docs_text)
    return sent_res

class TinyRAG:
    def __init__(self, config:RAGConfig) -> None:
        print("config: ", config)
        self.config = config
        self.searcher = Searcher(
            emb_model_id=config.emb_model_id,
            ranker_model_id=config.ranker_model_id,
            device=config.device,
            base_dir=config.base_dir
        )

        if self.config.model_type == "qwen2":
            self.llm:BaseLLM = Qwen2LLM(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
            # 传入最大生成长度（无条件设置）
            self.llm.max_new_tokens = getattr(self.config, "max_new_tokens", 128)
        elif self.config.model_type == "tinyllm":
            self.llm:BaseLLM = TinyLLM(
                model_id_key=config.llm_model_id,
                device=self.config.device
            )
            self.llm.max_new_tokens = getattr(self.config, "max_new_tokens", 128)
        else:
            raise "failed init LLM, the model type is [qwen2, tinyllm]"

    def build(self, docs: List[str]):
        """ 注意： 构建数据库需要很长时间
        """
        self.sent_split_model = SentenceSplitter(
            use_model=False, 
            sentence_size=self.config.sentence_size, 
            model_path=self.config.sent_split_model_id
        )
        logger.info("load sentence splitter model success! ")
        txt_list = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_item = {executor.submit(process_docs_text, item, self.sent_split_model): item for item in docs}
            
            for future in tqdm(as_completed(future_to_item), total=len(docs)):
                try:
                    sent_res = future.result()
                    sent_res = [item for item in sent_res if len(item) > 100]
                    txt_list.extend(sent_res)
                except Exception as exc:
                    logger.error(f"Generated an exception: {exc}")

        jsonl_list = [{"text": item} for item in txt_list]
        write_list_to_jsonl(jsonl_list, self.config.base_dir + "/split_sentence.jsonl")
        logger.info("split sentence success, all sentence number: ", len(txt_list))
        logger.info("build database ...... ")
        self.searcher.build_db(txt_list)
        logger.info("build database success, starting save .... ")
        self.searcher.save_db()
        logger.info("save database success!  ")

    def load(self):
        self.searcher.load_db()
        logger.info("search load database success!")

    def search(self, query: str, top_n:int = 3) -> str:
        # LLM的初次回答
        llm_result_txt = self.llm.generate(query)
        # 数据库检索的文本（用 query + 初次回答 做查询）
        search_content_list = self.searcher.search(query=query+llm_result_txt+query, top_n=top_n)
        content_list = [item[1] for item in search_content_list]

        # 基于 token 预算构造上下文，保证留足生成空间
        max_ctx = getattr(getattr(self.llm, "model", None), "config", None)
        max_ctx = getattr(max_ctx, "max_position_embeddings", 4096)
        max_new = getattr(self.llm, "max_new_tokens", 128)
        reserve = 8

        def _count_tokens(text: str) -> int:
            # 优先使用 LLM 的 tokenizer 做精确计数
            tok = getattr(self.llm, "tokenizer", None)
            if tok is not None:
                return len(tok.encode(text))
            # 回退到近似：按字符数估计
            return max(1, len(text) // 2)

        # 先构造固定部分（不含 context），计算其 token
        empty_context_prompt = RAG_PROMPT_TEMPALTE.format(
            context="",
            question=query,
            answer=llm_result_txt
        )
        fixed_tokens = _count_tokens(empty_context_prompt)
        budget = max(1, max_ctx - max_new - reserve - fixed_tokens)

        # 逐段累积 context，按预算裁剪
        used_contexts = []
        used_tokens = 0
        for piece in content_list:
            t = _count_tokens(piece + "\n")
            if used_tokens + t <= budget:
                used_contexts.append(piece)
                used_tokens += t
            else:
                # 尝试按预算截断当前片段
                remain = budget - used_tokens
                if remain <= 0:
                    break
                # 粗略截断（字符级），再校正
                approx = max(1, remain * 2)
                truncated = piece[:approx]
                # 迭代靠近预算边界
                while _count_tokens(truncated) > remain and len(truncated) > 10:
                    truncated = truncated[:-10]
                if _count_tokens(truncated) <= remain and len(truncated) > 0:
                    used_contexts.append(truncated)
                    used_tokens += _count_tokens(truncated)
                break

        context = "\n".join(used_contexts)
        prompt_text = RAG_PROMPT_TEMPALTE.format(
            context=context,
            question=query,
            answer=llm_result_txt
        )
        logger.info("prompt: {}".format(prompt_text))

        # 生成最终答案
        output = self.llm.generate(prompt_text)

        return output
        


