"""
File handling utilities for PDF extraction
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datetime import datetime

from .logger import get_logger

logger = get_logger("file_utils")


class FileHandler:
    """Handle file operations for PDF extraction"""
    
    def __init__(self, base_output_dir: Union[str, Path]):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
    
    def create_output_structure(self, pdf_name: str) -> Dict[str, Path]:
        """Create organized output directory structure for a PDF"""
        pdf_output_dir = self.base_output_dir / pdf_name.replace('.pdf', '')
        
        structure = {
            'base': pdf_output_dir,
            'tables': pdf_output_dir / 'tables',
            'text': pdf_output_dir / 'text',
            'images': pdf_output_dir / 'images',
            'forms': pdf_output_dir / 'forms',
            'reports': pdf_output_dir / 'reports',
            'raw_data': pdf_output_dir / 'raw_data'
        }
        
        # Create all directories
        for dir_path in structure.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output structure for {pdf_name}", 
                   context={'structure': {k: str(v) for k, v in structure.items()}})
        
        return structure
    
    def save_tables_to_excel(self, tables: List[pd.DataFrame], 
                           output_path: Path, sheet_names: List[str] = None) -> bool:
        """Save multiple tables to Excel with separate sheets"""
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                for i, table in enumerate(tables):
                    sheet_name = sheet_names[i] if sheet_names else f'Table_{i+1}'
                    # Clean sheet name (Excel limitations)
                    sheet_name = str(sheet_name)[:31].replace('/', '_').replace('\\', '_')
                    table.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Saved {len(tables)} tables to Excel", 
                       context={'file': str(output_path), 'tables': len(tables)})
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tables to Excel: {output_path}", 
                        exception=e, context={'tables_count': len(tables)})
            return False
    
    def save_tables_to_csv(self, tables: List[pd.DataFrame], 
                          output_dir: Path, prefix: str = "table") -> List[Path]:
        """Save tables as individual CSV files"""
        saved_files = []
        
        for i, table in enumerate(tables):
            csv_path = output_dir / f"{prefix}_{i+1}.csv"
            try:
                table.to_csv(csv_path, index=False)
                saved_files.append(csv_path)
                logger.debug(f"Saved table {i+1} to CSV", 
                           context={'file': str(csv_path), 'rows': len(table)})
            except Exception as e:
                logger.error(f"Failed to save table {i+1} to CSV", 
                           exception=e, context={'file': str(csv_path)})
        
        logger.info(f"Saved {len(saved_files)} tables to CSV files", 
                   context={'output_dir': str(output_dir)})
        return saved_files
    
    def save_json(self, data: Dict[str, Any], output_path: Path) -> bool:
        """Save data as JSON with proper formatting"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"Saved JSON data", context={'file': str(output_path)})
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON data", 
                        exception=e, context={'file': str(output_path)})
            return False
    
    def load_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON data", context={'file': str(file_path)})
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON data", 
                        exception=e, context={'file': str(file_path)})
            return None
    
    def save_text(self, text: str, output_path: Path) -> bool:
        """Save text content to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.debug(f"Saved text content", 
                        context={'file': str(output_path), 'length': len(text)})
            return True
        except Exception as e:
            logger.error(f"Failed to save text content", 
                        exception=e, context={'file': str(output_path)})
            return False
    
    def save_image(self, image_data: bytes, output_path: Path, 
                  image_format: str = 'PNG') -> bool:
        """Save image data to file"""
        try:
            with open(output_path, 'wb') as f:
                f.write(image_data)
            logger.debug(f"Saved image", 
                        context={'file': str(output_path), 'format': image_format, 
                                'size': len(image_data)})
            return True
        except Exception as e:
            logger.error(f"Failed to save image", 
                        exception=e, context={'file': str(output_path)})
            return False
    
    def create_extraction_report(self, pdf_name: str, results: Dict[str, Any], 
                               output_dir: Path) -> Path:
        """Create a comprehensive extraction report"""
        report_data = {
            'pdf_file': pdf_name,
            'extraction_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_pages': results.get('total_pages', 0),
                'tables_found': len(results.get('tables', [])),
                'images_found': len(results.get('images', [])),
                'form_fields_found': len(results.get('forms', {})),
                'text_length': len(results.get('text', ''))
            },
            'extraction_methods_used': results.get('methods_used', []),
            'accuracy_scores': results.get('accuracy_scores', {}),
            'processing_time': results.get('processing_time', {}),
            'detailed_results': results
        }
        
        # Save as JSON
        report_path = output_dir / f"extraction_report_{pdf_name.replace('.pdf', '')}.json"
        self.save_json(report_data, report_path)
        
        # Create readable text report
        text_report_path = output_dir / f"extraction_report_{pdf_name.replace('.pdf', '')}.txt"
        self._create_text_report(report_data, text_report_path)
        
        logger.info(f"Created extraction report", 
                   context={'json_report': str(report_path), 
                           'text_report': str(text_report_path)})
        
        return report_path
    
    def _create_text_report(self, report_data: Dict[str, Any], output_path: Path):
        """Create human-readable text report"""
        report_text = f"""
PDF EXTRACTION REPORT
====================

File: {report_data['pdf_file']}
Extraction Time: {report_data['extraction_timestamp']}

SUMMARY
-------
Total Pages: {report_data['summary']['total_pages']}
Tables Found: {report_data['summary']['tables_found']}
Images Found: {report_data['summary']['images_found']}
Form Fields Found: {report_data['summary']['form_fields_found']}
Text Length: {report_data['summary']['text_length']} characters

EXTRACTION METHODS USED
----------------------
{', '.join(report_data.get('extraction_methods_used', []))}

ACCURACY SCORES
--------------
"""
        
        for method, score in report_data.get('accuracy_scores', {}).items():
            report_text += f"{method}: {score:.2f}\n"
        
        report_text += f"""
PROCESSING TIME
--------------
"""
        for stage, time_taken in report_data.get('processing_time', {}).items():
            report_text += f"{stage}: {time_taken:.2f} seconds\n"
        
        self.save_text(report_text, output_path)
    
    def cleanup_temp_files(self, temp_dir: Path):
        """Clean up temporary files"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary files", 
                           context={'temp_dir': str(temp_dir)})
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files", 
                          exception=e, context={'temp_dir': str(temp_dir)})


def get_file_handler(output_dir: Union[str, Path]) -> FileHandler:
    """Get a configured file handler instance"""
    return FileHandler(output_dir)
