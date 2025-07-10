from app.core.config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingsFactory:
    @staticmethod
    def create():
        embeddings_provider = settings.EMBEDDINGS_PROVIDER.lower()

        if embeddings_provider == "openai":
            return OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
                model=settings.OPENAI_EMBEDDINGS_MODEL
            )
        elif embeddings_provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=settings.HUGGINGFACE_EMBEDDINGS_MODEL,
                model_kwargs={'device': settings.HUGGINGFACE_DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")
