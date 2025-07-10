from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import OllamaLLM
from langchain_community.llms import HuggingFacePipeline
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)

class LLMFactory:
    @staticmethod
    def create(
        provider: Optional[str] = None,
        temperature: float = 0,
        streaming: bool = True,
    ) -> BaseChatModel:
        """
        Create a LLM instance based on the provider
        """
        # If no provider specified, use the one from settings
        provider = provider or settings.CHAT_PROVIDER

        if provider.lower() == "openai":
            return ChatOpenAI(
                temperature=temperature,
                streaming=streaming,
                model=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE
            )
        elif provider.lower() == "deepseek":
            return ChatDeepSeek(
                temperature=temperature,
                streaming=streaming,
                model=settings.DEEPSEEK_MODEL,
                api_key=settings.DEEPSEEK_API_KEY,
                api_base=settings.DEEPSEEK_API_BASE
            )
        elif provider.lower() == "ollama":
            # Initialize Ollama model
            return OllamaLLM(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_API_BASE,
                temperature=temperature,
                streaming=streaming
            )
        elif provider.lower() == "huggingface_local":
            # Initialize Hugging Face local model
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
            
            # Hugging Face 토큰 명시적 설정
            if hasattr(settings, 'HUGGINGFACE_HUB_TOKEN') and settings.HUGGINGFACE_HUB_TOKEN:
                os.environ['HUGGINGFACE_HUB_TOKEN'] = settings.HUGGINGFACE_HUB_TOKEN
                from huggingface_hub import login
                login(settings.HUGGINGFACE_HUB_TOKEN)
                logger.info(f"[LLMFactory] Hugging Face token set: {settings.HUGGINGFACE_HUB_TOKEN[:10]}...")
            else:
                logger.warning("[LLMFactory] No Hugging Face token found in settings")
            
            logger.info(f"[LLMFactory] Loading Hugging Face tokenizer: {settings.HUGGINGFACE_LLM_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(settings.HUGGINGFACE_LLM_MODEL)
            
            # GPU 메모리 최적화 설정
            quantization_config = None
            if settings.HUGGINGFACE_LLM_LOAD_IN_4BIT:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_quant_type="nf4"
                )
            elif settings.HUGGINGFACE_LLM_LOAD_IN_8BIT:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info(f"[LLMFactory] Loading Hugging Face model: {settings.HUGGINGFACE_LLM_MODEL} (device={settings.HUGGINGFACE_LLM_DEVICE}, quant={quantization_config})")
            model = AutoModelForCausalLM.from_pretrained(
                settings.HUGGINGFACE_LLM_MODEL,
                device_map=settings.HUGGINGFACE_LLM_DEVICE,
                quantization_config=quantization_config,
                torch_dtype="auto" if settings.HUGGINGFACE_LLM_DEVICE == "cuda" else None
            )
            logger.info(f"[LLMFactory] Creating pipeline for model: {settings.HUGGINGFACE_LLM_MODEL}")
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=settings.HUGGINGFACE_LLM_MAX_LENGTH,
                temperature=temperature,
                device=settings.HUGGINGFACE_LLM_DEVICE
            )
            logger.info(f"[LLMFactory] HuggingFacePipeline created successfully for model: {settings.HUGGINGFACE_LLM_MODEL}")
            return HuggingFacePipeline(pipeline=pipe)
        # Add more providers here as needed
        # elif provider.lower() == "anthropic":
        #     return ChatAnthropic(...)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")