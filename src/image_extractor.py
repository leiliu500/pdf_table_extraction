"""
Image extraction module for PDF files
"""
import time
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .config.settings import IMAGE_EXTRACTION_SETTINGS
from .utils.logger import get_logger

logger = get_logger("image_extractor")


@dataclass
class ExtractedImage:
    """Represents an extracted image from a PDF"""
    filename: str
    format: str
    width: int
    height: int
    page_number: int
    image_data: bytes
    bbox: Optional[List[float]] = None
    size_bytes: int = 0
    dpi: Optional[Tuple[int, int]] = None


@dataclass
class ImageExtractionResult:
    """Result of image extraction from a single method"""
    method: str
    images: List[ExtractedImage]
    metadata: Dict[str, Any]
    processing_time: float
    extraction_successful: bool
    error_message: Optional[str] = None


class PyMuPDFImageExtractor:
    """Image extraction using PyMuPDF"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PYMUPDF_AVAILABLE and settings.get('extract_images', True)
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available for image extraction")
    
    def extract_images(self, pdf_path: str, output_dir: Path) -> ImageExtractionResult:
        """Extract images using PyMuPDF"""
        if not self.enabled:
            return ImageExtractionResult(
                method="pymupdf",
                images=[],
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="PyMuPDF not available or image extraction disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting PyMuPDF image extraction", context={'pdf': pdf_path})
        
        try:
            doc = fitz.open(pdf_path)
            extracted_images = []
            metadata = {'pages_with_images': [], 'total_images': 0, 'formats': {}}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                if image_list:
                    page_images = []
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Check image dimensions
                            if (pix.width < self.settings.get('min_width', 50) or 
                                pix.height < self.settings.get('min_height', 50)):
                                pix = None
                                continue
                            
                            # Convert to appropriate format
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_format = "PNG"
                                img_data = pix.tobytes("png")
                            else:  # CMYK
                                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                img_format = "PNG"
                                img_data = pix1.tobytes("png")
                                pix1 = None
                            
                            # Create filename
                            filename = f"page_{page_num + 1}_img_{img_index + 1}.{img_format.lower()}"
                            
                            # Save image file
                            image_path = output_dir / filename
                            with open(image_path, 'wb') as f:
                                f.write(img_data)
                            
                            # Create ExtractedImage object
                            extracted_img = ExtractedImage(
                                filename=filename,
                                format=img_format,
                                width=pix.width,
                                height=pix.height,
                                page_number=page_num + 1,
                                image_data=img_data,
                                size_bytes=len(img_data),
                                dpi=pix.xres if hasattr(pix, 'xres') else None
                            )
                            
                            extracted_images.append(extracted_img)
                            page_images.append(filename)
                            
                            # Update metadata
                            metadata['formats'][img_format] = metadata['formats'].get(img_format, 0) + 1
                            
                            logger.debug(f"Extracted image {filename} from page {page_num + 1}")
                            
                            pix = None
                        
                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}", 
                                         exception=e)
                    
                    if page_images:
                        metadata['pages_with_images'].append({
                            'page': page_num + 1,
                            'image_count': len(page_images),
                            'images': page_images
                        })
            
            doc.close()
            
            metadata['total_images'] = len(extracted_images)
            processing_time = time.time() - start_time
            
            logger.info(f"PyMuPDF image extraction completed", 
                       context={
                           'total_images': len(extracted_images),
                           'pages_with_images': len(metadata['pages_with_images']),
                           'processing_time': processing_time,
                           'formats': list(metadata['formats'].keys())
                       })
            
            return ImageExtractionResult(
                method="pymupdf",
                images=extracted_images,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PyMuPDF image extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return ImageExtractionResult(
                method="pymupdf",
                images=[],
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )


class PDFPlumberImageExtractor:
    """Image extraction using pdfplumber"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.enabled = PDFPLUMBER_AVAILABLE and settings.get('extract_images', True)
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available for image extraction")
    
    def extract_images(self, pdf_path: str, output_dir: Path) -> ImageExtractionResult:
        """Extract images using pdfplumber"""
        if not self.enabled:
            return ImageExtractionResult(
                method="pdfplumber",
                images=[],
                metadata={},
                processing_time=0.0,
                extraction_successful=False,
                error_message="pdfplumber not available or image extraction disabled"
            )
        
        start_time = time.time()
        logger.info(f"Starting pdfplumber image extraction", context={'pdf': pdf_path})
        
        try:
            extracted_images = []
            metadata = {'pages_with_images': [], 'total_images': 0}
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Get images on this page
                    if hasattr(page, 'images'):
                        images = page.images
                        page_images = []
                        
                        for img_index, img in enumerate(images):
                            try:
                                # Extract image properties
                                bbox = [img.get('x0', 0), img.get('y0', 0), 
                                       img.get('x1', 0), img.get('y1', 0)]
                                width = int(img.get('width', 0))
                                height = int(img.get('height', 0))
                                
                                # Check minimum size requirements
                                if (width < self.settings.get('min_width', 50) or 
                                    height < self.settings.get('min_height', 50)):
                                    continue
                                
                                # Create filename
                                filename = f"page_{page_num}_img_{img_index + 1}_pdfplumber.png"
                                
                                # Note: pdfplumber doesn't directly provide image data
                                # We can only get metadata about images
                                extracted_img = ExtractedImage(
                                    filename=filename,
                                    format="PNG",
                                    width=width,
                                    height=height,
                                    page_number=page_num,
                                    image_data=b'',  # No actual image data available
                                    bbox=bbox,
                                    size_bytes=0
                                )
                                
                                extracted_images.append(extracted_img)
                                page_images.append(filename)
                                
                                logger.debug(f"Identified image {filename} on page {page_num}")
                            
                            except Exception as e:
                                logger.warning(f"Failed to process image {img_index} from page {page_num}", 
                                             exception=e)
                        
                        if page_images:
                            metadata['pages_with_images'].append({
                                'page': page_num,
                                'image_count': len(page_images),
                                'images': page_images
                            })
            
            metadata['total_images'] = len(extracted_images)
            processing_time = time.time() - start_time
            
            logger.info(f"pdfplumber image extraction completed", 
                       context={
                           'total_images': len(extracted_images),
                           'pages_with_images': len(metadata['pages_with_images']),
                           'processing_time': processing_time
                       })
            
            # Note: This method only identifies images, doesn't extract actual data
            logger.warning("pdfplumber can only identify images, not extract image data")
            
            return ImageExtractionResult(
                method="pdfplumber",
                images=extracted_images,
                metadata=metadata,
                processing_time=processing_time,
                extraction_successful=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"pdfplumber image extraction failed", exception=e,
                        context={'pdf': pdf_path, 'processing_time': processing_time})
            
            return ImageExtractionResult(
                method="pdfplumber",
                images=[],
                metadata={},
                processing_time=processing_time,
                extraction_successful=False,
                error_message=str(e)
            )


class ImageExtractor:
    """Main image extraction class that orchestrates multiple extraction methods"""
    
    def __init__(self, settings: Dict[str, Any] = None):
        self.settings = settings or IMAGE_EXTRACTION_SETTINGS
        
        # Initialize extractors
        self.extractors = {}
        
        if PYMUPDF_AVAILABLE and self.settings.get('extract_images', True):
            self.extractors['pymupdf'] = PyMuPDFImageExtractor(self.settings)
        
        if PDFPLUMBER_AVAILABLE and self.settings.get('extract_images', True):
            self.extractors['pdfplumber'] = PDFPlumberImageExtractor(self.settings)
        
        logger.info(f"Initialized image extractor", 
                   context={'enabled_methods': list(self.extractors.keys())})
    
    def extract_all_images(self, pdf_path: str, output_dir: Path) -> Dict[str, Any]:
        """Extract images using all available methods"""
        logger.trace_pdf_processing(pdf_path, "image_extraction_start")
        
        start_time = time.time()
        results = {}
        extraction_results = {}
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all extraction methods
        for method_name, extractor in self.extractors.items():
            logger.info(f"Running {method_name} image extraction")
            
            try:
                result = extractor.extract_images(pdf_path, output_dir)
                extraction_results[method_name] = result
                results[method_name] = result.images
                
                logger.info(f"{method_name} image extraction completed", 
                           context={
                               'images_found': len(result.images),
                               'successful': result.extraction_successful,
                               'processing_time': result.processing_time
                           })
                
            except Exception as e:
                logger.error(f"{method_name} image extraction failed", exception=e)
                extraction_results[method_name] = ImageExtractionResult(
                    method=method_name,
                    images=[],
                    metadata={},
                    processing_time=0.0,
                    extraction_successful=False,
                    error_message=str(e)
                )
                results[method_name] = []
        
        # Determine best method
        best_method = self._determine_best_method(extraction_results)
        
        # Combine unique images
        combined_images = self._combine_images(results)
        
        total_time = time.time() - start_time
        
        final_results = {
            'images_by_method': results,
            'extraction_results': extraction_results,
            'combined_images': combined_images,
            'best_method': best_method,
            'total_processing_time': total_time,
            'summary': {
                'total_images_combined': len(combined_images),
                'successful_methods': len([r for r in extraction_results.values() if r.extraction_successful]),
                'methods_used': list(self.extractors.keys()),
                'output_directory': str(output_dir)
            }
        }
        
        logger.trace_pdf_processing(
            pdf_path, "image_extraction_complete", 
            {'total_images': len(combined_images),
             'best_method': best_method}
        )
        
        return final_results
    
    def _combine_images(self, results: Dict[str, List[ExtractedImage]]) -> List[ExtractedImage]:
        """Combine images from multiple methods, removing duplicates"""
        combined = []
        seen_images = set()
        
        # Prioritize PyMuPDF as it extracts actual image data
        method_priority = ['pymupdf', 'pdfplumber']
        
        for method in method_priority:
            if method in results:
                for image in results[method]:
                    # Use page number and approximate size to identify duplicates
                    image_key = (image.page_number, image.width, image.height)
                    if image_key not in seen_images:
                        combined.append(image)
                        seen_images.add(image_key)
        
        # Add any remaining images from other methods
        for method, images in results.items():
            if method not in method_priority:
                for image in images:
                    image_key = (image.page_number, image.width, image.height)
                    if image_key not in seen_images:
                        combined.append(image)
                        seen_images.add(image_key)
        
        logger.info(f"Combined images from multiple methods", 
                   context={'total_combined': len(combined), 'methods': list(results.keys())})
        
        return combined
    
    def _determine_best_method(self, extraction_results: Dict[str, ImageExtractionResult]) -> str:
        """Determine the best extraction method based on results"""
        method_scores = {}
        
        for method, result in extraction_results.items():
            if not result.extraction_successful:
                method_scores[method] = 0.0
                continue
            
            # Scoring factors
            image_count_score = min(len(result.images) / 10, 1.0)
            
            # Bonus for methods that extract actual image data
            data_quality_score = 1.0 if any(img.size_bytes > 0 for img in result.images) else 0.5
            
            speed_score = max(0, 1.0 - (result.processing_time / 10))
            
            # Weighted average
            total_score = (
                image_count_score * 0.4 +
                data_quality_score * 0.5 +
                speed_score * 0.1
            )
            
            method_scores[method] = total_score
        
        if not method_scores:
            return "none"
        
        best_method = max(method_scores, key=method_scores.get)
        
        logger.info(f"Best image extraction method determined", 
                   context={'method': best_method, 'scores': method_scores})
        
        return best_method
    
    def get_best_images(self, extraction_results: Dict[str, Any]) -> List[ExtractedImage]:
        """Get the best images from extraction results"""
        # Return combined images if available (usually best)
        if 'combined_images' in extraction_results:
            return extraction_results['combined_images']
        
        best_method = extraction_results.get('best_method', 'none')
        
        if best_method == 'none' or best_method not in extraction_results['images_by_method']:
            # Fallback: return images from the method with most images
            images_by_method = extraction_results['images_by_method']
            if not images_by_method:
                return []
            
            best_method = max(images_by_method, key=lambda k: len(images_by_method[k]))
        
        return extraction_results['images_by_method'].get(best_method, [])
