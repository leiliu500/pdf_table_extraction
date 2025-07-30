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

  # Enable debug logging
  python main.py -i pdf/file.pdf -o output/ --log-level DEBUG

  # Disable validation for faster processing
  python main.py -i pdf/file.pdf -o output/ --no-validation
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input PDF file or directory containing PDF files"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
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
            for category, score in confidence.items():
                print(f"  {category.capitalize()}: {score*100:.1f}%")
            
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
    
    # Validate mutually exclusive options
    if args.tables_only and args.texts_only:
        print("Error: --tables-only and --texts-only cannot be used together")
        sys.exit(1)
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Load custom configuration if provided
    custom_settings = None
    if args.config:
        custom_settings = load_custom_config(args.config)
        if custom_settings is None:
            print(f"Error: Failed to load configuration from {args.config}")
            sys.exit(1)
    
    # Initialize extractor
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
        
    except Exception as e:
        print(f"Error: Failed to initialize PDF extractor: {e}")
        sys.exit(1)
    
    # Perform extraction
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
            
            results = extractor.extract_all(pattern=args.pattern)
            
            if not results:
                print(f"No PDF files found matching pattern '{args.pattern}'")
                sys.exit(1)
            
            print_results_summary(results)
    
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error("Extraction failed", exception=e)
        print(f"Error: Extraction failed: {e}")
        sys.exit(1)
    
    print(f"\nExtraction completed. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
