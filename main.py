#!/usr/bin/env python3
"""
Main entry point for PDF table extraction tool
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pdf_extractor import PDFExtractor
from src.utils.logger import get_logger

logger = get_logger("main")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Extract tables, text, forms, and images from PDF files with high accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single PDF file
  python main.py -i pdf/304-Cedar-Street/1.pdf -o output/

  # Extract from all PDFs in directory
  python main.py -i pdf/304-Cedar-Street/ -o output/

  # Extract only tables
  python main.py -i pdf/file.pdf -o output/ --tables-only

  # Extract only text
  python main.py -i pdf/file.pdf -o output/ --texts-only

  # Extract only forms
  python main.py -i pdf/file.pdf -o output/ --forms-only

  # Extract only images
  python main.py -i pdf/file.pdf -o output/ --images-only

  # Enable debug logging
  python main.py -i pdf/file.pdf -o output/ --log-level DEBUG

  # Disable validation for faster processing
  python main.py -i pdf/file.pdf -o output/ --no-validation
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=False,  # Not required for RAG query mode
        help="Input PDF file or directory containing PDF files"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=False,  # Not required for RAG query mode
        help="Output directory for extraction results"
    )
    
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Extract only tables (faster processing)"
    )
    
    parser.add_argument(
        "--texts-only",
        action="store_true",
        help="Extract only text content (faster processing)"
    )
    
    parser.add_argument(
        "--forms-only",
        action="store_true",
        help="Extract only form fields (faster processing)"
    )
    
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Extract only images (faster processing)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable accuracy validation (faster processing)"
    )
    
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern for directory processing (default: *.pdf)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file (JSON)"
    )
    
    # RAG System Arguments
    parser.add_argument(
        "--rag-process",
        action="store_true",
        help="Process PDF through RAG system for question answering"
    )
    
    parser.add_argument(
        "--rag-query",
        type=str,
        help="Query processed PDFs using RAG system"
    )
    
    parser.add_argument(
        "--rag-stats",
        action="store_true", 
        help="Show RAG system statistics"
    )
    
    return parser


def load_custom_config(config_path: str) -> Optional[dict]:
    """Load custom configuration from JSON file"""
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded custom configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}", exception=e)
        return None


def print_results_summary(results):
    """Print a summary of extraction results"""
    if isinstance(results, list):
        # Multiple files processed
        total_files = len(results)
        successful = len([r for r in results if r.extraction_successful])
        failed = total_files - successful
        
        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {total_files}")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {failed}")
        
        if successful > 0:
            print(f"\nSuccessful files:")
            for result in results:
                if result.extraction_successful:
                    pdf_name = Path(result.pdf_path).name
                    tables = result.tables.get('summary', {}).get('total_tables_found', 0)
                    text_len = result.text.get('summary', {}).get('best_text_length', 0)
                    forms = result.forms.get('summary', {}).get('total_fields_combined', 0)
                    images = result.images.get('summary', {}).get('total_images_combined', 0)
                    confidence = result.confidence_scores.get('overall', 0) * 100
                    
                    print(f"  • {pdf_name}: {tables} tables, {text_len} chars text, "
                          f"{forms} form fields, {images} images (confidence: {confidence:.1f}%)")
        
        if failed > 0:
            print(f"\nFailed files:")
            for result in results:
                if not result.extraction_successful:
                    pdf_name = Path(result.pdf_path).name
                    errors = "; ".join(result.error_messages[:2])  # Show first 2 errors
                    print(f"  • {pdf_name}: {errors}")
    
    else:
        # Single file processed
        result = results
        pdf_name = Path(result.pdf_path).name
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION RESULTS: {pdf_name}")
        print(f"{'='*60}")
        print(f"Status: {'SUCCESS' if result.extraction_successful else 'FAILED'}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
        if result.extraction_successful:
            # Table results
            table_summary = result.tables.get('summary', {})
            print(f"\nTABLES:")
            print(f"  Found: {table_summary.get('total_tables_found', 0)}")
            if 'best_method' in result.tables:
                print(f"  Best method: {result.tables['best_method']}")
            
            # Text results
            text_summary = result.text.get('summary', {})
            print(f"\nTEXT:")
            print(f"  Length: {text_summary.get('best_text_length', 0)} characters")
            if 'best_method' in result.text:
                print(f"  Best method: {result.text['best_method']}")
            
            # Form results
            form_summary = result.forms.get('summary', {})
            print(f"\nFORMS:")
            print(f"  Fields found: {form_summary.get('total_fields_combined', 0)}")
            if 'best_method' in result.forms:
                print(f"  Best method: {result.forms['best_method']}")
            
            # Image results
            image_summary = result.images.get('summary', {})
            print(f"\nIMAGES:")
            print(f"  Found: {image_summary.get('total_images_combined', 0)}")
            if 'best_method' in result.images:
                print(f"  Best method: {result.images['best_method']}")
            
            # Confidence scores
            confidence = result.confidence_scores
            print(f"\nCONFIDENCE SCORES:")
            if isinstance(confidence, dict):
                for category, score in confidence.items():
                    print(f"  {category.capitalize()}: {score*100:.1f}%")
            else:
                print(f"  Overall: {confidence*100:.1f}%")
            
            # Output files
            if result.output_files:
                print(f"\nOUTPUT FILES:")
                for category, files in result.output_files.items():
                    print(f"  {category.capitalize()}: {len(files)} files")
        
        else:
            print(f"\nERRORS:")
            for error in result.error_messages:
                print(f"  • {error}")


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle RAG query mode (doesn't require input file)
    if args.rag_query:
        try:
            import asyncio
            from src.rag import query_pdf_documents
            
            print(f"Querying RAG system...")
            print(f"Question: {args.rag_query}")
            print(f"{'='*60}")
            
            result = asyncio.run(query_pdf_documents(args.rag_query))
            
            if result['status'] == 'success':
                print(f"Answer: {result['answer']}")
                print(f"\nConfidence: {result['confidence_score']*100:.1f}%")
                print(f"Processing time: {result['processing_time']:.2f}s")
                print(f"Sources consulted: {result['retrieved_chunks']} chunks")
                
                # Show citations if available
                citations = result.get('citations', [])
                if citations:
                    print(f"\nSources:")
                    for citation in citations:
                        filename = citation.get('filename', 'Unknown')
                        page = citation.get('page_number', 'N/A')
                        content_type = citation.get('content_type', 'text')
                        similarity = citation.get('similarity_score', 0.0)
                        print(f"  [{citation['index']}] {filename} (Page {page}, {content_type}) - Relevance: {similarity*100:.1f}%")
                
            elif result['status'] == 'no_results':
                print(f"No relevant information found for your query.")
                print("Try rephrasing your question or make sure documents are processed first.")
                
            else:
                print(f"Query failed: {result.get('error', 'Unknown error')}")
                
        except ImportError as e:
            print(f"RAG system dependencies not available: {e}")
            print("Please install RAG dependencies: pip install -r requirements.txt")
        except Exception as e:
            print(f"Query failed: {e}")
        
        return
    
    # Handle RAG stats mode
    if args.rag_stats:
        try:
            import asyncio
            from src.rag import create_rag_pipeline
            
            async def get_rag_stats():
                print("RAG System Statistics")
                print("="*60)
                
                pipeline = await create_rag_pipeline()
                stats = await pipeline.get_pipeline_stats()
                await pipeline.close()
                return stats
            
            stats = asyncio.run(get_rag_stats())
            
            # Processing stats
            proc_stats = stats.get('processing_stats', {})
            print(f"Processing Statistics:")
            print(f"  • Documents processed: {proc_stats.get('documents_processed', 0)}")
            print(f"  • Total chunks created: {proc_stats.get('total_chunks_created', 0)}")
            print(f"  • Embeddings generated: {proc_stats.get('embeddings_generated', 0)}")
            print(f"  • Queries processed: {proc_stats.get('queries_processed', 0)}")
            
            processing_times = proc_stats.get('processing_times', [])
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                print(f"  • Average processing time: {avg_time:.2f}s")
            
            # Database stats
            db_stats = stats.get('database_stats', {})
            if db_stats:
                print(f"\nDatabase Statistics:")
                print(f"  • Total documents: {db_stats.get('total_documents', 0)}")
                print(f"  • Total embeddings: {db_stats.get('total_embeddings', 0)}")
                print(f"  • Average confidence: {db_stats.get('average_confidence', 0)*100:.1f}%")
                
                chunks_by_type = db_stats.get('chunks_by_type', {})
                if chunks_by_type:
                    print(f"  • Chunks by type: {', '.join(f'{k}({v})' for k, v in chunks_by_type.items())}")
            
            # Ollama stats
            ollama_stats = stats.get('ollama_stats', {})
            if ollama_stats:
                print(f"\nOllama Performance:")
                print(f"  • Embedding model: {ollama_stats.get('embedding_model', 'Unknown')}")
                print(f"  • LLM model: {ollama_stats.get('llm_model', 'Unknown')}")
                print(f"  • Using fallback embeddings: {ollama_stats.get('using_fallback_embeddings', False)}")
                
                emb_stats = ollama_stats.get('embedding_stats', {})
                if emb_stats:
                    print(f"  • Embedding requests: {emb_stats.get('total_requests', 0)}")
                    print(f"  • Average embedding time: {emb_stats.get('average_time', 0):.2f}s")
                
                llm_stats = ollama_stats.get('llm_stats', {})
                if llm_stats:
                    print(f"  • LLM requests: {llm_stats.get('total_requests', 0)}")
                    print(f"  • Average response time: {llm_stats.get('average_time', 0):.2f}s")
            
            # Config info
            config = stats.get('pipeline_config', {})
            if config:
                print(f"\nConfiguration:")
                print(f"  • Chunk size: {config.get('chunk_size', 0)}")
                print(f"  • Similarity top-k: {config.get('similarity_top_k', 0)}")
                print(f"  • Accuracy threshold: {config.get('accuracy_threshold', 0)*100:.1f}%")
                
        except ImportError as e:
            print(f"RAG system dependencies not available: {e}")
            print("Please install RAG dependencies: pip install -r requirements.txt")
        except Exception as e:
            print(f"Failed to get RAG stats: {e}")
        
        return
    
    # Check for RAG query or stats modes that don't need input/output
    if args.rag_query or args.rag_stats:
        # These modes don't require input/output arguments
        pass
    else:
        # All other modes require input and output
        if not args.input:
            print("Error: Input file/directory (-i/--input) is required for extraction modes")
            sys.exit(1)
        if not args.output:
            print("Error: Output directory (-o/--output) is required for extraction modes")
            sys.exit(1)
    
    # Validate mutually exclusive options
    exclusive_options = [args.tables_only, args.texts_only, args.forms_only, args.images_only, args.rag_process]
    if sum(exclusive_options) > 1:
        print("Error: --tables-only, --texts-only, --forms-only, --images-only, and --rag-process cannot be used together")
        sys.exit(1)
    
    # Validate input path (only if provided)
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            sys.exit(1)
    else:
        input_path = None
    
    # Load custom configuration if provided
    custom_settings = None
    if args.config:
        custom_settings = load_custom_config(args.config)
        if custom_settings is None:
            print(f"Error: Failed to load configuration from {args.config}")
            sys.exit(1)
    
    # Initialize extractor (only if needed for extraction modes)
    extractor = None
    if input_path and args.output:
        try:
            extractor = PDFExtractor(
                input_dir=input_path if input_path.is_dir() else input_path.parent,
                output_dir=args.output,
                log_level=args.log_level,
                enable_validation=not args.no_validation,
                custom_settings=custom_settings
            )
            
            print(f"Initialized PDF extractor")
            print(f"Input: {input_path}")
            print(f"Output: {args.output}")
            print(f"Validation: {'Enabled' if not args.no_validation else 'Disabled'}")
            print(f"Log level: {args.log_level}")
            print()
            
        except Exception as e:
            print(f"Error initializing PDF extractor: {e}")
            sys.exit(1)
    
    # Perform extraction
    if extractor and input_path:
        try:
            if input_path.is_file():
                # Single file
                if args.tables_only:
                    print(f"\nExtracting tables only from: {input_path.name}")
                    table_results = extractor.extract_tables_only(input_path)
                    
                    # Print table-only summary
                    print(f"\n{'='*60}")
                    print(f"TABLE EXTRACTION RESULTS: {input_path.name}")
                    print(f"{'='*60}")
                    print(f"Tables found: {table_results['summary']['total_tables_found']}")
                    print(f"Best method: {table_results.get('best_method', 'N/A')}")
                    print(f"Processing time: {table_results.get('total_processing_time', 0):.2f} seconds")
                    
                    if table_results['summary']['total_tables_found'] > 0:
                        print(f"\nMethods used:")
                        for method in table_results['summary']['methods_used']:
                            method_results = table_results['extraction_results'].get(method)
                            if method_results and method_results.extraction_successful:
                                tables_count = len(method_results.tables)
                                avg_confidence = sum(method_results.confidence_scores) / len(method_results.confidence_scores) if method_results.confidence_scores else 0
                                print(f"  • {method}: {tables_count} tables (confidence: {avg_confidence*100:.1f}%)")
                    
                    print(f"\nExtraction completed. Results saved to: {args.output}")
                    return
            
            elif args.texts_only:
                print(f"\nExtracting text only from: {input_path.name}")
                text_results = extractor.extract_texts_only(input_path)
                
                # Print text-only summary
                print(f"\n{'='*60}")
                print(f"TEXT EXTRACTION RESULTS: {input_path.name}")
                print(f"{'='*60}")
                
                best_text = extractor.text_extractor.get_best_text(text_results)
                text_length = len(best_text) if best_text else 0
                print(f"Text extracted: {text_length:,} characters")
                print(f"Best method: {text_results.get('best_method', 'N/A')}")
                print(f"Processing time: {text_results.get('total_processing_time', 0):.2f} seconds")
                
                if text_length > 0:
                    print(f"\nMethods used:")
                    for method in text_results['summary']['methods_used']:
                        method_results = text_results['extraction_results'].get(method)
                        if method_results and method_results.extraction_successful:
                            method_text_length = len(method_results.text) if method_results.text else 0
                            confidence_score = method_results.confidence_score if hasattr(method_results, 'confidence_score') else 0
                            print(f"  • {method}: {method_text_length:,} characters (confidence: {confidence_score*100:.1f}%)")
                    
                    # Show text preview
                    preview_length = min(200, text_length)
                    preview = best_text[:preview_length].replace('\n', ' ').strip()
                    if text_length > preview_length:
                        preview += "..."
                    print(f"\nText preview: {preview}")
                
                print(f"\nExtraction completed. Results saved to: {args.output}")
                return

            elif args.forms_only:
                print(f"\nExtracting forms only from: {input_path.name}")
                form_results = extractor.extract_forms_only(input_path)
                
                # Print form-only summary
                print(f"\n{'='*60}")
                print(f"FORM EXTRACTION RESULTS: {input_path.name}")
                print(f"{'='*60}")
                
                best_fields = extractor.form_extractor.get_best_fields(form_results)
                fields_count = len(best_fields) if best_fields else 0
                print(f"Form fields found: {fields_count}")
                print(f"Best method: {form_results.get('best_method', 'N/A')}")
                print(f"Processing time: {form_results.get('total_processing_time', 0):.2f} seconds")
                
                if fields_count > 0:
                    print(f"\nMethods used:")
                    for method in form_results['summary']['methods_used']:
                        method_results = form_results['extraction_results'].get(method)
                        if method_results and method_results.extraction_successful:
                            method_fields_count = len(method_results.form_fields) if hasattr(method_results, 'form_fields') and method_results.form_fields else 0
                            confidence_score = method_results.confidence_score if hasattr(method_results, 'confidence_score') else 0
                            print(f"  • {method}: {method_fields_count} fields (confidence: {confidence_score*100:.1f}%)")
                    
                    # Show fields preview
                    if isinstance(best_fields, list) and len(best_fields) > 0:
                        print(f"\nForm fields preview:")
                        preview_count = min(5, len(best_fields))
                        for i, field in enumerate(best_fields[:preview_count]):
                            field_name = field.get('name', f'Field_{i+1}') if isinstance(field, dict) else f'Field_{i+1}'
                            field_value = field.get('value', 'N/A') if isinstance(field, dict) else str(field)
                            if len(str(field_value)) > 50:
                                field_value = str(field_value)[:50] + "..."
                            print(f"  • {field_name}: {field_value}")
                        if len(best_fields) > preview_count:
                            print(f"  ... and {len(best_fields) - preview_count} more fields")
                
                print(f"\nExtraction completed. Results saved to: {args.output}")
                return

            elif args.images_only:
                print(f"\nExtracting images only from: {input_path.name}")
                image_results = extractor.extract_images_only(input_path)
                
                print(f"\n{'='*60}")
                print(f"IMAGE EXTRACTION RESULTS: {input_path.name}")
                print(f"{'='*60}")
                
                total_images = 0
                best_method = "N/A"
                processing_time = 0
                
                # Get summary from results
                if image_results and 'summary' in image_results:
                    total_images = image_results['summary'].get('total_images_combined', 0)
                    best_method = image_results.get('best_method', 'N/A')
                    processing_time = image_results.get('total_processing_time', 0)
                
                print(f"Images found: {total_images}")
                print(f"Best method: {best_method}")
                print(f"Processing time: {processing_time:.2f} seconds")
                
                if total_images > 0:
                    print(f"\nMethods used:")
                    for method in image_results['summary']['methods_used']:
                        method_results = image_results['extraction_results'].get(method)
                        if method_results and method_results.extraction_successful:
                            images_count = len(method_results.images)
                            
                            # Calculate average OCR confidence from images with OCR data
                            ocr_confidences = [img.ocr_confidence for img in method_results.images 
                                             if hasattr(img, 'ocr_confidence') and img.ocr_confidence is not None]
                            avg_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0
                            
                            if avg_confidence > 0:
                                print(f"  • {method}: {images_count} images (avg OCR confidence: {avg_confidence*100:.1f}%)")
                            else:
                                print(f"  • {method}: {images_count} images")
                    
                    # Show OCR summary if available
                    combined_images = image_results.get('combined_images', [])
                    images_with_text = [img for img in combined_images 
                                      if hasattr(img, 'ocr_text') and img.ocr_text and len(img.ocr_text.strip()) > 3]
                    
                    if images_with_text:
                        avg_ocr_confidence = sum(img.ocr_confidence for img in images_with_text if img.ocr_confidence) / len(images_with_text)
                        print(f"\nOCR Results:")
                        print(f"  • {len(images_with_text)} images contain readable text")
                        print(f"  • Average OCR confidence: {avg_ocr_confidence*100:.1f}%")
                        print(f"  • Total text extracted: {sum(len(img.ocr_text) for img in images_with_text)} characters")
                
                print(f"\nExtraction completed. Results saved to: {args.output}")
                return

            elif args.rag_process:
                print(f"\nProcessing PDF through RAG system: {input_path.name}")
                
                try:
                    import asyncio
                    from src.rag import process_pdf_with_rag
                    
                    # Process PDF through RAG pipeline
                    result = asyncio.run(process_pdf_with_rag(str(input_path)))
                    
                    print(f"\n{'='*60}")
                    print(f"RAG PROCESSING RESULTS: {input_path.name}")
                    print(f"{'='*60}")
                    
                    if result['status'] == 'success':
                        print(f"✓ Document processed successfully")
                        print(f"  • Document ID: {result['document_id']}")
                        print(f"  • Processing time: {result['processing_time']:.2f}s")
                        
                        # Show extraction summary
                        extraction_summary = result.get('extraction_summary', {})
                        print(f"\nExtraction Summary:")
                        for content_type, stats in extraction_summary.items():
                            count = stats.get('count', 0)
                            avg_conf = stats.get('average_confidence', 0.0)
                            print(f"  • {content_type.title()}: {count} items (avg confidence: {avg_conf*100:.1f}%)")
                        
                        # Show chunk summary
                        chunk_summary = result.get('chunk_summary', {})
                        if chunk_summary:
                            print(f"\nChunk Summary:")
                            print(f"  • Total chunks: {chunk_summary.get('total_chunks', 0)}")
                            print(f"  • Average confidence: {chunk_summary.get('average_confidence', 0.0)*100:.1f}%")
                            print(f"  • Average chunk length: {chunk_summary.get('average_chunk_length', 0):.0f} characters")
                            
                            type_dist = chunk_summary.get('content_type_distribution', {})
                            if type_dist:
                                print(f"  • Content types: {', '.join(f'{k}({v})' for k, v in type_dist.items())}")
                        
                        # Show accuracy metrics
                        accuracy_metrics = result.get('accuracy_metrics', {})
                        if accuracy_metrics:
                            print(f"\nAccuracy Metrics:")
                            for metric, value in accuracy_metrics.items():
                                print(f"  • {metric.replace('_', ' ').title()}: {value*100:.1f}%")
                        
                        print(f"\n✓ Document ready for querying!")
                        print(f"Use: python main.py --rag-query \"Your question here\"")
                        
                    elif result['status'] == 'already_processed':
                        print(f"ℹ Document already processed")
                        print(f"  • Document ID: {result['document_id']}")
                        doc_info = result.get('document_info', {})
                        if doc_info:
                            print(f"  • Originally processed: {doc_info.get('processing_date', 'Unknown')}")
                            print(f"  • Total chunks: {doc_info.get('total_chunks', 0)}")
                    
                    else:
                        print(f"✗ Processing failed: {result.get('error', 'Unknown error')}")
                        return
                
                except ImportError as e:
                    print(f"✗ RAG system dependencies not available: {e}")
                    print("Please install RAG dependencies: pip install -r requirements.txt")
                    return
                except Exception as e:
                    print(f"✗ RAG processing failed: {e}")
                    return
                
                return

            else:
                print(f"\nExtracting all content from: {input_path.name}")
                result = extractor.extract_pdf(input_path)
                print_results_summary(result)
        
        else:
            # Directory
            print(f"\nExtracting from directory: {input_path}")
            print(f"Pattern: {args.pattern}")
            
            if args.tables_only:
                print("Note: --tables-only flag ignored for directory processing")
            elif args.texts_only:
                print("Note: --texts-only flag ignored for directory processing")
            elif args.forms_only:
                print("Note: --forms-only flag ignored for directory processing")
            elif args.images_only:
                print("Note: --images-only flag ignored for directory processing")
            
            results = extractor.extract_all(pattern=args.pattern)
            
            if not results:
                print(f"No PDF files found matching pattern '{args.pattern}'")
                sys.exit(1)
            
            print_results_summary(results)
        
        except KeyboardInterrupt:
            print("\nExtraction interrupted by user")
            sys.exit(1)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error("Extraction failed", exception=e)
            print(f"Error: Extraction failed: {e}")
            sys.exit(1)
        
        print(f"\nExtraction completed. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
