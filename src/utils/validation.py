"""
Validation utilities for accuracy checking and comparison
"""
import difflib
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from .logger import get_logger

logger = get_logger("validation")


@dataclass
class ValidationResult:
    """Result of a validation check"""
    score: float
    passed: bool
    details: Dict[str, Any]
    method: str
    confidence: float = 0.0


class TableValidator:
    """Validate extracted table data accuracy"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
    
    def compare_table_extractions(self, table1: pd.DataFrame, table2: pd.DataFrame, 
                                method1: str, method2: str) -> ValidationResult:
        """Compare two table extractions to identify differences"""
        details = {}
        
        # Shape comparison
        shape_match = table1.shape == table2.shape
        details['shape_match'] = shape_match
        details['table1_shape'] = table1.shape
        details['table2_shape'] = table2.shape
        
        if not shape_match:
            logger.warning(f"Table shape mismatch between {method1} and {method2}", 
                          context=details)
            return ValidationResult(
                score=0.0, 
                passed=False, 
                details=details, 
                method=f"{method1}_vs_{method2}"
            )
        
        # Content comparison
        try:
            # Normalize DataFrames before comparison
            # Reset indexes and use generic column names
            norm1 = table1.copy().reset_index(drop=True)
            norm2 = table2.copy().reset_index(drop=True)
            
            # Use generic column names to avoid column label mismatch
            norm1.columns = [f'col_{i}' for i in range(len(norm1.columns))]
            norm2.columns = [f'col_{i}' for i in range(len(norm2.columns))]
            
            # Convert to string for comparison
            str1 = norm1.fillna('').astype(str)
            str2 = norm2.fillna('').astype(str)
            
            # Cell-by-cell comparison
            matches = (str1 == str2).sum().sum()
            total_cells = str1.size
            similarity = matches / total_cells if total_cells > 0 else 0.0
            
            # Identify differences
            diff_cells = []
            for i in range(str1.shape[0]):
                for j in range(str1.shape[1]):
                    if str1.iloc[i, j] != str2.iloc[i, j]:
                        diff_cells.append({
                            'row': i,
                            'col': j,
                            f'{method1}_value': str1.iloc[i, j],
                            f'{method2}_value': str2.iloc[i, j]
                        })
            
            details.update({
                'similarity_score': similarity,
                'total_cells': total_cells,
                'matching_cells': matches,
                'different_cells': len(diff_cells),
                'differences': diff_cells[:10]  # First 10 differences
            })
            
            passed = similarity >= self.confidence_threshold
            
            logger.trace_extraction_comparison(
                method1, method2, similarity, 
                {'different_cells': len(diff_cells)}
            )
            
            return ValidationResult(
                score=similarity,
                passed=passed,
                details=details,
                method=f"{method1}_vs_{method2}",
                confidence=similarity
            )
            
        except Exception as e:
            logger.error(f"Error comparing tables from {method1} and {method2}", 
                        exception=e)
            return ValidationResult(
                score=0.0,
                passed=False,
                details={'error': str(e)},
                method=f"{method1}_vs_{method2}"
            )
    
    def validate_table_structure(self, table: pd.DataFrame, 
                                expected_structure: Dict[str, Any] = None) -> ValidationResult:
        """Validate table structure against expected format"""
        details = {}
        score = 1.0
        
        # Basic structure checks
        details['has_data'] = not table.empty
        details['row_count'] = len(table)
        details['column_count'] = len(table.columns)
        
        if table.empty:
            score = 0.0
            details['issues'] = ['Table is empty']
        
        # Check for expected structure if provided
        if expected_structure:
            issues = []
            
            if 'min_rows' in expected_structure:
                if len(table) < expected_structure['min_rows']:
                    issues.append(f"Too few rows: {len(table)} < {expected_structure['min_rows']}")
                    score *= 0.8
            
            if 'min_columns' in expected_structure:
                if len(table.columns) < expected_structure['min_columns']:
                    issues.append(f"Too few columns: {len(table.columns)} < {expected_structure['min_columns']}")
                    score *= 0.8
            
            if 'required_columns' in expected_structure:
                missing_cols = set(expected_structure['required_columns']) - set(table.columns)
                if missing_cols:
                    issues.append(f"Missing required columns: {missing_cols}")
                    score *= 0.7
            
            details['issues'] = issues
        
        # Data quality checks
        null_percentage = table.isnull().sum().sum() / (len(table) * len(table.columns))
        details['null_percentage'] = null_percentage
        
        if null_percentage > 0.5:  # More than 50% null values
            score *= 0.6
            details.setdefault('issues', []).append(f"High null percentage: {null_percentage:.2%}")
        
        passed = score >= self.confidence_threshold
        
        return ValidationResult(
            score=score,
            passed=passed,
            details=details,
            method="structure_validation",
            confidence=score
        )
    
    def calculate_extraction_confidence(self, extraction_results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for multiple extraction methods"""
        if not extraction_results:
            return 0.0
        
        method_scores = []
        
        for result in extraction_results:
            # Factors that affect confidence
            factors = {
                'table_count': min(result.get('table_count', 0) / 10, 1.0),  # Normalize to 0-1
                'has_structure': 1.0 if result.get('has_clear_structure', False) else 0.5,
                'extraction_success': 1.0 if result.get('extraction_successful', False) else 0.0
            }
            
            # Weighted average
            weights = {'table_count': 0.3, 'has_structure': 0.4, 'extraction_success': 0.3}
            score = sum(factors[k] * weights[k] for k in factors)
            method_scores.append(score)
        
        # Overall confidence is the average of all methods
        overall_confidence = sum(method_scores) / len(method_scores)
        
        logger.trace_accuracy_validation(
            "extraction_confidence", overall_confidence, 
            overall_confidence >= self.confidence_threshold,
            {'method_scores': method_scores, 'methods_count': len(extraction_results)}
        )
        
        return overall_confidence


class TextValidator:
    """Validate extracted text accuracy"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
    
    def compare_text_extractions(self, text1: str, text2: str, 
                               method1: str, method2: str) -> ValidationResult:
        """Compare two text extractions"""
        # Calculate similarity using sequence matcher
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # Find differences
        diff = list(difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            fromfile=method1,
            tofile=method2,
            n=3
        ))
        
        details = {
            'similarity_score': similarity,
            'text1_length': len(text1),
            'text2_length': len(text2),
            'diff_lines': len(diff),
            'diff_preview': ''.join(diff[:20])  # First 20 lines of diff
        }
        
        passed = similarity >= self.confidence_threshold
        
        logger.trace_extraction_comparison(
            method1, method2, similarity, 
            {'text_length_diff': abs(len(text1) - len(text2))}
        )
        
        return ValidationResult(
            score=similarity,
            passed=passed,
            details=details,
            method=f"{method1}_vs_{method2}",
            confidence=similarity
        )


class AccuracyValidator:
    """Main validator class combining all validation methods"""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.table_validator = TableValidator(confidence_threshold)
        self.text_validator = TextValidator(confidence_threshold)
    
    def validate_extraction_results(self, results: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Run comprehensive validation on extraction results"""
        validation_results = {}
        
        # Validate tables if multiple extraction methods were used
        tables = results.get('tables', {})
        if len(tables) > 1:
            methods = list(tables.keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    if tables[method1] and tables[method2]:
                        for j, (table1, table2) in enumerate(zip(tables[method1], tables[method2])):
                            if isinstance(table1, pd.DataFrame) and isinstance(table2, pd.DataFrame):
                                key = f"table_comparison_{method1}_{method2}_{j}"
                                validation_results[key] = self.table_validator.compare_table_extractions(
                                    table1, table2, method1, method2
                                )
        
        # Validate text if multiple extraction methods were used
        texts = results.get('text', {})
        if len(texts) > 1:
            methods = list(texts.keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    if texts[method1] and texts[method2]:
                        key = f"text_comparison_{method1}_{method2}"
                        validation_results[key] = self.text_validator.compare_text_extractions(
                            texts[method1], texts[method2], method1, method2
                        )
        
        # Calculate overall accuracy score
        if validation_results:
            scores = [v.score for v in validation_results.values()]
            overall_score = sum(scores) / len(scores)
            overall_passed = overall_score >= self.confidence_threshold
            
            validation_results['overall'] = ValidationResult(
                score=overall_score,
                passed=overall_passed,
                details={'individual_scores': scores, 'methods_compared': len(validation_results)},
                method="overall_validation",
                confidence=overall_score
            )
            
            logger.trace_accuracy_validation(
                "overall_extraction", overall_score, overall_passed,
                {'validations_performed': len(validation_results)}
            )
        
        return validation_results


def get_validator(confidence_threshold: float = 0.8) -> AccuracyValidator:
    """Get a configured validator instance"""
    return AccuracyValidator(confidence_threshold)
