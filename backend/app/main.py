import logging

from app.api.api_v1.api import api_router
from app.api.openapi.api import router as openapi_router
from app.core.config import settings
from app.core.minio import init_minio
from app.startup.migarate import DatabaseMigrator
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(openapi_router, prefix="/openapi")


@app.on_event("startup")
async def startup_event():
    # Initialize MinIO
    init_minio()
    # Run database migrations
    migrator = DatabaseMigrator(settings.get_database_url)
    migrator.run_migrations()
    
    # Pre-load Hugging Face models
    print("Pre-loading Hugging Face models...")
    try:
        # Pre-load LLM model if using Hugging Face
        if settings.CHAT_PROVIDER.lower() == "huggingface_local":
            print(f"Pre-loading LLM model: {settings.HUGGINGFACE_LLM_MODEL}")
            from app.services.llm.llm_factory import LLMFactory
            llm = LLMFactory.create()
            print("LLM model loaded successfully!")
        
        # Pre-load Embedding model if using Hugging Face
        if settings.EMBEDDINGS_PROVIDER.lower() == "huggingface":
            print(f"Pre-loading Embedding model: {settings.HUGGINGFACE_EMBEDDINGS_MODEL}")
            from app.services.embedding.embedding_factory import EmbeddingsFactory
            embeddings = EmbeddingsFactory.create()
            print("Embedding model loaded successfully!")
        
        print("All Hugging Face models pre-loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to pre-load models: {e}")
        print("Models will be loaded on first request.")


@app.get("/")
def root():
    return {"message": "Welcome to RAG Web UI API"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }
