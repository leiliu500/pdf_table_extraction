"""
RAG System Configuration
"""

# Ollama Configuration
OLLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',
    'embedding_model': 'nomic-embed-text',
    'llm_model': 'llama3.1:8b',
    'temperature': 0.1,
    'top_k': 40,
    'top_p': 0.9,
    'max_tokens': 2048,
    'timeout': 120  # Increased timeout for complex queries
}

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'pdf_rag',
    'user': 'ttoulliu2002',
    'password': 'aA101@math',
    'pool_size': 10,
    'max_overflow': 20
}

# Vector Database Configuration
VECTOR_CONFIG = {
    'dimension': 768,  # nomic-embed-text embedding dimension
    'similarity_threshold': 0.3,  # Lower threshold for better recall
    'max_results': 10,
    'index_type': 'ivfflat',
    'index_lists': 100
}

# RAG Pipeline Configuration
RAG_CONFIG = {
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'min_chunk_size': 100,
    'max_chunk_size': 2000,
    'similarity_top_k': 5,
    'rerank_top_k': 3,
    'context_window': 4000,
    'answer_length': 500,
    'enable_reranking': True,
    'enable_citation': True,
    'accuracy_threshold': 0.8
}

# Document Processing Configuration
DOCUMENT_CONFIG = {
    'enable_multimodal': True,
    'process_tables': True,
    'process_images': True,
    'process_forms': True,
    'enable_ocr': True,
    'accuracy_validation': True,
    'confidence_threshold': 0.7,
    'max_file_size_mb': 100,
    'supported_formats': ['.pdf']
}

# Accuracy and Quality Configuration
ACCURACY_CONFIG = {
    'extraction_confidence_threshold': 0.5,  # Lower for better recall
    'embedding_quality_threshold': 0.7,
    'answer_confidence_threshold': 0.6,
    'enable_accuracy_logging': True,
    'enable_validation': True,
    'accuracy_metrics': ['precision', 'recall', 'f1_score'],
    'quality_checks': ['completeness', 'consistency', 'relevance']
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}',
    'rotation': '10 MB',
    'retention': '1 week',
    'accuracy_log_file': 'logs/accuracy.log',
    'performance_log_file': 'logs/performance.log',
    'rag_log_file': 'logs/rag_pipeline.log'
}
