"""
BOQ Pre-Processor Module

This module handles the initial analysis and extraction of BOQ (Bill of Quantities) 
data from Excel files. It identifies potential headers, analyzes worksheet structure,
and extracts BOQ tables with AI assistance.
"""

import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter, column_index_from_string
from openpyxl.utils.cell import coordinate_from_string
import re
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import requests
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings about column duplication in console
import warnings
warnings.filterwarnings('ignore', message='DataFrame columns are not unique')

@dataclass
class CellInfo:
    """Detailed cell information"""
    value: Any
    coordinate: str
    is_merged: bool
    merged_range: Optional[str]
    has_formula: bool
    formula: Optional[str]
    is_wrapped: bool
    font_info: Dict
    fill_info: Dict

@dataclass
class WorksheetAnalysis:
    """Comprehensive worksheet analysis"""
    name: str
    dimensions: Tuple[int, int]  # (max_row, max_col)
    frozen_panes: Optional[str]
    merged_cells: List[str]
    suspected_headers: List[Tuple[int, List[str], float]]  # (row, headers, confidence)
    boq_regions: List[Dict]
    data_density: float
    has_wrapped_text: bool
    ai_analysis: Optional[Dict]

@dataclass
class BOQTable:
    """Enhanced BOQ table structure"""
    worksheet_name: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    headers: List[str]
    data: pd.DataFrame
    confidence_score: float
    ai_validated: bool
    extraction_notes: List[str]
    cell_details: List[CellInfo]

@dataclass
class FileAnalysis:
    """Complete file analysis"""
    filename: str
    file_size: int
    worksheets: List[WorksheetAnalysis]
    total_boq_tables: int
    extraction_summary: Dict
    processing_time: float

class AdvancedBOQExtractor:
    def __init__(self, gemini_api_key: str = None):
        """Initialize the extractor with Gemini AI integration using direct HTTP requests"""
        
        # Enhanced BOQ patterns with variations
        self.boq_patterns = {
            'sl_no': [
                r'(?i)(sl\.?\s*no\.?|s\.?\s*no\.?|serial\s*no\.?|item\s*no\.?|sr\.?\s*no\.?)',
                r'(?i)(sequence|order|number|#)',
            ],
            'code': [
                r'(?i)(code|item\s*code|work\s*code|activity\s*code)',
                r'(?i)(ref\.?\s*no\.?|reference)',
            ],
            'description': [
                r'(?i)(description|item\s*description|particulars|work\s*description)',
                r'(?i)(activity|task|work\s*item|specification)',
            ],
            'unit': [
                r'(?i)(unit|uom|u\.?o\.?m\.?|units)',
                r'(?i)(measure|measurement)',
            ],
            'quantity': [
                r'(?i)(qty|quantity|cumulative|overall\s*qty|total\s*qty)',
                r'(?i)(volume|count|number)',
            ],
            'rate': [
                r'(?i)(rate|unit\s*rate|net\s*rate|basic\s*rate)',
                r'(?i)(price|cost\s*per\s*unit)',
            ],
            'amount': [
                r'(?i)(amount|total\s*amount|value|total\s*value)',
                r'(?i)(cost|total\s*cost)',
            ],
            'grade': [
                r'(?i)(grade|grade\s*of\s*conc|concrete\s*grade)',
                r'(?i)(class|type|category)',
            ],
            'cement': [
                r'(?i)(cement\s*co-?eff|cement\s*coefficient|cement)',
            ],
            'remarks': [
                r'(?i)(remarks|notes|comments|observations)',
                r'(?i)(specification|remark)',
            ]
        }
        
        # FIXED: Initialize Gemini AI with proper API key handling
        self.api_key = gemini_api_key
        
        # FIXED: Remove the placeholder check that was causing issues
        if not self.api_key or self.api_key.strip() == "":
            logger.warning("Gemini API key not provided. AI features will be disabled.")
            self.ai_enabled = False
        else:
            # Updated API URL to use the correct endpoint and model
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            self.cache = {}  # Simple cache to avoid duplicate API calls
            
            # API call counter
            self.api_call_count = 0
            
            self.ai_enabled = True
            logger.info(f"✅ Gemini AI initialized successfully for preprocessor (key length: {len(self.api_key)} chars)")
    
    def validate_excel_file(self, file_path: str) -> bool:
        """Validate if the file is a proper Excel file"""
        try:
            # Check file extension
            valid_extensions = ['.xlsx', '.xls', '.xlsm']
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                logger.warning(f"File {file_path} doesn't have a valid Excel extension")
                return False
            
            # Try to open with openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            wb.close()
            return True
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {str(e)}")
            return False
    
    def analyze_cell_details(self, ws, cell_coord: str) -> CellInfo:
        """Get detailed information about a specific cell"""
        cell = ws[cell_coord]
        
        # Check if cell is part of merged range
        is_merged = False
        merged_range = None
        for merged_cell in ws.merged_cells.ranges:
            if cell.coordinate in merged_cell:
                is_merged = True
                merged_range = str(merged_cell)
                break
        
        # Get font and fill information
        font_info = {
            'name': cell.font.name if cell.font else None,
            'size': cell.font.size if cell.font else None,
            'bold': cell.font.bold if cell.font else False,
            'italic': cell.font.italic if cell.font else False
        }
        
        fill_info = {
            'pattern_type': cell.fill.patternType if cell.fill else None,
            'start_color': str(cell.fill.start_color.rgb) if cell.fill and cell.fill.start_color else None
        }
        
        return CellInfo(
            value=cell_coord,
            coordinate=cell_coord,
            is_merged=is_merged,
            merged_range=merged_range,
            has_formula=cell.data_type == 'f',
            formula=cell.value if cell.data_type == 'f' else None,
            is_wrapped=cell.alignment.wrap_text if cell.alignment else False,
            font_info=font_info,
            fill_info=fill_info
        )
    
    def handle_merged_cells(self, ws, df: pd.DataFrame) -> pd.DataFrame:
        """Handle merged cells by propagating values"""
        logger.info(f"Processing {len(list(ws.merged_cells.ranges))} merged cell ranges")
        
        for merged_range in ws.merged_cells.ranges:
            # Get the top-left cell value
            min_row, min_col, max_row, max_col = merged_range.bounds
            top_left_value = ws.cell(min_row, min_col).value
            
            # Propagate value to all cells in the merged range
            for row in range(min_row - 1, max_row):  # -1 because pandas is 0-indexed
                for col in range(min_col - 1, max_col):
                    if row < len(df) and col < len(df.columns):
                        if pd.isna(df.iloc[row, col]) or df.iloc[row, col] == '':
                            df.iloc[row, col] = top_left_value
        
        return df
    
    def detect_wrapped_text_content(self, ws) -> List[str]:
        """Detect cells with wrapped text and extract their content"""
        wrapped_content = []
        
        for row in ws.iter_rows():
            for cell in row:
                if cell.alignment and cell.alignment.wrap_text and cell.value:
                    wrapped_content.append(str(cell.value))
        
        logger.info(f"Found {len(wrapped_content)} cells with wrapped text")
        return wrapped_content
    
    def analyze_worksheet_structure(self, ws, sheet_name: str, file_path: str) -> WorksheetAnalysis:
        """Comprehensive analysis of worksheet structure"""
        logger.info(f"Analyzing worksheet: {sheet_name}")
        
        # Basic dimensions
        max_row = ws.max_row
        max_col = ws.max_column
        
        # Frozen panes
        frozen_panes = str(ws.freeze_panes) if ws.freeze_panes else None
        
        # Merged cells
        merged_cells = [str(merged_range) for merged_range in ws.merged_cells.ranges]
        
        # Check for wrapped text
        has_wrapped_text = any(
            cell.alignment and cell.alignment.wrap_text 
            for row in ws.iter_rows() for cell in row if cell.alignment
        )
        
        # Calculate data density
        total_cells = max_row * max_col
        filled_cells = sum(1 for row in ws.iter_rows() for cell in row if cell.value is not None)
        data_density = filled_cells / total_cells if total_cells > 0 else 0
        
        # Read data for analysis with better error handling
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine='openpyxl')
            df = self.handle_merged_cells(ws, df)
        except Exception as e:
            logger.error(f"Error reading worksheet {sheet_name}: {str(e)}")
            # Create empty dataframe as fallback
            df = pd.DataFrame()
        
        # Find suspected headers
        suspected_headers = self.find_all_potential_headers(df, sheet_name) if not df.empty else []
        
        # Use AI to select the best header from top 2 candidates
        selected_header = self.ai_select_best_header(suspected_headers, sheet_name)
        
        # Only create BOQ region for the AI-selected header
        boq_regions = []
        if selected_header:
            header_row, headers, confidence = selected_header
            
            # Find data boundaries for the selected header
            start_row, end_row = self.extract_table_boundaries(df, header_row)
            
            if end_row >= start_row:
                region = {
                    'header_row': header_row,
                    'start_row': start_row,
                    'end_row': end_row,
                    'headers': headers,
                    'confidence': confidence,
                    'data_rows': end_row - start_row + 1,
                    'ai_selected': True  # Mark as AI-selected
                }
                boq_regions.append(region)
                logger.info(f"AI-selected header created 1 BOQ region with {region['data_rows']} data rows")
        
        analysis = WorksheetAnalysis(
            name=sheet_name,
            dimensions=(max_row, max_col),
            frozen_panes=frozen_panes,
            merged_cells=merged_cells,
            suspected_headers=suspected_headers,  # Keep all candidates for reference
            boq_regions=boq_regions,  # Only 1 region max
            data_density=data_density,
            has_wrapped_text=has_wrapped_text,
            ai_analysis=None
        )
        
        # Simple AI analysis if enabled
        if self.ai_enabled and boq_regions:
            analysis.ai_analysis = self.ai_analyze_worksheet(df, analysis)
        
        return analysis
    
    def is_likely_data_row(self, row_values: List[str]) -> bool:
        """Better check to distinguish data rows from header rows"""
        data_indicators = 0
        total_non_empty = 0
        
        for value in row_values:
            value_str = str(value).strip()
            if value_str and value_str.lower() != 'nan':
                total_non_empty += 1
                
                # Better data row indicators
                if re.match(r'^[A-Z]\s*\d+$', value_str):  # "A 2", "B 1" 
                    data_indicators += 2  # Strong indicator
                elif re.match(r'^\d+\.?\d*$', value_str):  # Pure numbers
                    data_indicators += 1
                elif re.match(r'^[A-Z]{2,}\d+$', value_str):  # "PCC003"
                    data_indicators += 2  # Strong indicator
                elif len(value_str) > 80:  # Very long descriptions
                    data_indicators += 1
                elif any(word in value_str.lower() for word in ['excavation', 'concrete', 'steel', 'masonry']):
                    data_indicators += 1  # Construction terms
        
        # Higher threshold for data row detection
        return (data_indicators / total_non_empty) > 0.6 if total_non_empty > 0 else False
    
    def find_all_potential_headers(self, df: pd.DataFrame, sheet_name: str) -> List[Tuple[int, List[str], float]]:
        """Better header detection with realistic confidence scores"""
        potential_headers = []
        
        # Search through first 25 rows for headers
        search_range = min(25, len(df))
        
        for idx in range(search_range):
            row_data = df.iloc[idx].astype(str).fillna('')
            
            if row_data.str.strip().eq('').all():  # Skip completely empty rows
                continue
            
            # Get non-empty values for analysis
            non_empty_values = [cell for cell in row_data if cell and str(cell).strip() and str(cell).strip() != 'nan']
            
            if len(non_empty_values) < 2:  # Headers should have at least 2 columns
                continue
            
            # Better filter for data rows
            if self.is_likely_data_row(non_empty_values):
                logger.debug(f"Skipping row {idx+1} - looks like data: {non_empty_values[:3]}")
                continue
            
            # Better pattern matching
            score = 0
            matched_patterns = []
            
            # Check each cell against BOQ patterns
            for cell_value in row_data:
                cell_str = str(cell_value).strip()
                if cell_str and cell_str != 'nan':
                    for pattern_name, patterns in self.boq_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, cell_str):
                                score += 1
                                matched_patterns.append(pattern_name)
                                break
            
            # More realistic confidence calculation
            non_empty_cells = len(non_empty_values)
            
            if non_empty_cells > 0 and score > 0:
                # Base confidence - never exceeds 0.8 without AI
                confidence = min(score / non_empty_cells, 0.8)
                
                # More conservative boosts
                unique_patterns = len(set(matched_patterns))
                if unique_patterns >= 4:
                    confidence *= 1.2
                elif unique_patterns >= 3:
                    confidence *= 1.1
                elif unique_patterns >= 2:
                    confidence *= 1.05
                
                # Boost for critical BOQ patterns
                critical_patterns = ['sl_no', 'description', 'unit', 'quantity', 'rate', 'amount']
                if any(pattern in matched_patterns for pattern in critical_patterns):
                    confidence *= 1.1
                
                # Cap at 0.9 (never 1.0 without AI validation)
                confidence = min(confidence, 0.9)
                
                if confidence > 0.25:  # Lower threshold but more realistic scoring
                    headers = [str(cell).strip() for cell in row_data]
                    potential_headers.append((idx, headers, confidence))
                    logger.info(f"Header candidate at row {idx+1}: confidence={confidence:.3f}, patterns={list(set(matched_patterns))}")
        
        # Smart deduplication: only one high-confidence header per region
        potential_headers = self.smart_deduplicate_headers(potential_headers)
        
        # Sort by confidence
        potential_headers.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(potential_headers)} potential header rows in {sheet_name}")
        
        return potential_headers
    
    def smart_deduplicate_headers(self, headers: List[Tuple[int, List[str], float]]) -> List[Tuple[int, List[str], float]]:
        """Remove conflicting headers - only keep one high-confidence header per region"""
        if not headers:
            return headers
        
        # Sort by confidence first
        headers.sort(key=lambda x: x[2], reverse=True)
        
        filtered = []
        used_rows = set()
        
        for row_idx, header_list, confidence in headers:
            # Check if this row conflicts with already selected headers
            conflicts = False
            for used_row in used_rows:
                if abs(row_idx - used_row) <= 5:  # Within 5 rows = same region
                    conflicts = True
                    break
            
            if not conflicts:
                filtered.append((row_idx, header_list, confidence))
                used_rows.add(row_idx)
        
        return filtered
    
    def call_gemini_api(self, prompt: str) -> Optional[Dict]:
        """Make a direct HTTP request to Gemini API"""
        if not self.api_key:
            return None
            
        # Prepare the API request
        url = f"{self.base_url}?key={self.api_key}"
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            logger.info(f"Making Gemini API call #{self.api_call_count + 1}")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            # Update tracking variable
            self.api_call_count += 1
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    text_response = result['candidates'][0]['content']['parts'][0]['text']
                    logger.info(f"Raw AI response: {text_response[:200]}...")  # Log first 200 chars
                    
                    # Clean up the response - remove markdown formatting
                    cleaned_response = text_response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()
                    
                    try:
                        # Try to parse as JSON
                        ai_result = json.loads(cleaned_response)
                        logger.info(f"AI analysis successful: is_boq={ai_result.get('is_boq', 'unknown')}")
                        return ai_result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse AI response as JSON: {str(e)}")
                        logger.warning(f"Response text: {cleaned_response[:500]}")
                        # Return a default structure if parsing fails
                        return {
                            "is_boq": True,  # Assume it's BOQ if we can't parse
                            "confidence": 0.5,
                            "boq_type": "construction",
                            "data_quality": "medium",
                            "extraction_recommendation": "extract",
                            "notes": "AI response parsing failed, using default values"
                        }
                else:
                    logger.error("No candidates in Gemini response")
                    return None
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Gemini API call: {str(e)}")
            return None
    
    def ai_select_best_header(self, suspected_headers: List[Tuple[int, List[str], float]], sheet_name: str) -> Optional[Tuple[int, List[str], float]]:
        """Use AI to select the best header from top 2 candidates"""
        if not self.ai_enabled or not suspected_headers:
            return suspected_headers[0] if suspected_headers else None
        
        # Take top 2 candidates for AI validation
        top_candidates = suspected_headers[:2]
        
        # Prepare candidates for AI
        candidates_info = []
        for i, (row_idx, headers, conf) in enumerate(top_candidates):
            candidates_info.append({
                "option": i + 1,
                "row": row_idx + 1,
                "confidence": conf,
                "headers": headers[:6]  # First 6 headers only
            })
        
        prompt = f"""Select the best BOQ header from these {len(candidates_info)} candidates:

WORKSHEET: {sheet_name}
CANDIDATES:
{json.dumps(candidates_info, indent=2)}

Select the BEST header based on:
1. Standard BOQ columns (S.No, Description, Unit, Quantity, Rate, Amount)
2. Proper header format (not data rows or titles)
3. Most complete BOQ structure

Respond ONLY with JSON:
{{
  "selected_option": 1,
  "confidence": 0.95,
  "reasoning": "Option 1 has complete BOQ structure with standard columns"
}}

If NONE are valid, set selected_option to 0.
JSON only, no other text."""
        
        ai_result = self.call_gemini_api(prompt)
        
        if ai_result:
            selected_option = ai_result.get('selected_option', 0)
            if 1 <= selected_option <= len(top_candidates):
                selected_header = top_candidates[selected_option - 1]
                logger.info(f"AI selected header option {selected_option} for {sheet_name}: Row {selected_header[0]+1}")
                return selected_header
            else:
                logger.warning(f"AI rejected all headers for {sheet_name}")
                return None
        else:
            logger.warning(f"AI header selection failed for {sheet_name}, using top candidate")
            return top_candidates[0]  # Fallback to highest confidence
    
    def ai_analyze_worksheet(self, df: pd.DataFrame, analysis: WorksheetAnalysis) -> Dict:
        """Simple validation of the final selected header"""
        # This is now just a simple confirmation, not used for validation
        return {"is_boq": True, "confidence": 0.8, "data_quality": "good"}
    
    def clean_headers(self, headers: List[str]) -> List[str]:
        """Clean and standardize header names, handling duplicates"""
        cleaned = []
        seen_headers = {}
        
        for i, header in enumerate(headers):
            if pd.isna(header):
                clean_header = f"Column_{i+1}"
            else:
                # Remove extra whitespace and newlines, limit length
                clean_header = re.sub(r'\s+', ' ', str(header).strip())
                # Limit header length to 50 characters for readability
                if len(clean_header) > 50:
                    clean_header = clean_header[:47] + "..."
            
            # Handle duplicates by adding a number suffix
            original_header = clean_header
            counter = 1
            while clean_header in seen_headers:
                clean_header = f"{original_header}_{counter}"
                counter += 1
            
            seen_headers[clean_header] = True
            cleaned.append(clean_header)
        
        return cleaned
    
    def extract_table_boundaries(self, df: pd.DataFrame, header_row: int) -> Tuple[int, int]:
        """Extract table start and end rows, continuing until the true end of data"""
        start_row = header_row + 1
        end_row = -1  # Default to -1 if no data is found below the header

        # Scan from the bottom up to find the last row with any data
        for idx in range(len(df) - 1, start_row - 1, -1):
            if not df.iloc[idx].isna().all():
                end_row = idx
                break
                
        return start_row, end_row

    def extract_boq_table(self, file_path: str, ws_analysis: WorksheetAnalysis, region: Dict) -> Optional[BOQTable]:
        """Extract a BOQ table from a specific region"""
        try:
            # Read worksheet data
            df = pd.read_excel(file_path, sheet_name=ws_analysis.name, header=None, engine='openpyxl')
            
            # Handle merged cells
            wb = openpyxl.load_workbook(file_path, data_only=False)
            ws = wb[ws_analysis.name]
            df = self.handle_merged_cells(ws, df)
            
            # Extract the specific region
            header_row = region['header_row']
            start_row = region['start_row']
            end_row = region['end_row']
            
            headers = region['headers']
            data_df = df.iloc[start_row:end_row+1].copy()
            
            # Clean up data
            clean_headers = self.clean_headers(headers)
            data_df.columns = clean_headers[:len(data_df.columns)]
            data_df = data_df.dropna(how='all')  # Remove completely empty rows
            
            # Collect cell details for verification
            cell_details = []
            for row_idx in range(header_row, min(header_row + 10, end_row + 1)):
                for col_idx in range(min(10, len(clean_headers))):
                    try:
                        cell_coord = f"{get_column_letter(col_idx + 1)}{row_idx + 1}"
                        cell_info = self.analyze_cell_details(ws, cell_coord)
                        cell_details.append(cell_info)
                    except:
                        continue
            
            # AI validation is now header-level, not worksheet-level
            ai_validated = region.get('ai_selected', False)
            
            extraction_notes = [
                f"Header selected by AI from {len(ws_analysis.suspected_headers)} candidates" if ai_validated else f"Header confidence: {region['confidence']:.3f}",
                f"Contains {len(data_df)} data rows",
                f"AI header selection: {'YES' if ai_validated else 'NO'}"
            ]
            
            boq_table = BOQTable(
                worksheet_name=ws_analysis.name,
                start_row=header_row,
                start_col=0,
                end_row=end_row,
                end_col=len(clean_headers)-1,
                headers=clean_headers,
                data=data_df,
                confidence_score=region['confidence'],
                ai_validated=ai_validated,  # Now means "AI selected this header"
                extraction_notes=extraction_notes,
                cell_details=cell_details
            )
            
            logger.info(f"Successfully extracted BOQ table from {ws_analysis.name}: {len(data_df)} rows, AI selected: {ai_validated}")
            return boq_table
            
        except Exception as e:
            logger.error(f"Error extracting BOQ table from {ws_analysis.name}: {str(e)}")
            return None
    
    def process_excel_file(self, file_path: str) -> FileAnalysis:
        """Process a complete Excel file and return analysis results"""
        start_time = time.time()
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Load workbook with error handling
        try:
            wb = openpyxl.load_workbook(file_path, data_only=False)
        except Exception as e:
            logger.error(f"Failed to load workbook: {str(e)}")
            raise Exception(f"Cannot open Excel file. File may be corrupted or password protected: {str(e)}")
        
        # Analyze each worksheet
        worksheet_analyses = []
        boq_tables = []
        
        for sheet_name in wb.sheetnames:
            try:
                ws = wb[sheet_name]
                ws_analysis = self.analyze_worksheet_structure(ws, sheet_name, file_path)
                worksheet_analyses.append(ws_analysis)
                
                # Each worksheet now has max 1 BOQ region (AI-selected header)
                if ws_analysis.boq_regions:
                    region = ws_analysis.boq_regions[0]  # Only 1 region per worksheet
                    if region['data_rows'] > 0:
                        boq_table = self.extract_boq_table(file_path, ws_analysis, region)
                        if boq_table:
                            boq_tables.append(boq_table)
                            logger.info(f"Created 1 BOQ table for {sheet_name}")
                    else:
                        logger.warning(f"Skipping BOQ region in {sheet_name} as it contains no data rows.")
                    
            except Exception as e:
                logger.error(f"Error processing worksheet {sheet_name}: {str(e)}")
                continue
        
        processing_time = time.time() - start_time
        
        # Create summary
        extraction_summary = {
            'total_worksheets': len(worksheet_analyses),
            'worksheets_with_boq': len([ws for ws in worksheet_analyses if ws.boq_regions]),
            'total_boq_tables': len(boq_tables),
            'high_confidence_tables': len([bt for bt in boq_tables if bt.confidence_score > 0.7]),
            'ai_validated_tables': len([bt for bt in boq_tables if bt.ai_validated]) if self.ai_enabled else 0,
            'processing_time_seconds': processing_time
        }
        
        file_analysis = FileAnalysis(
            filename=os.path.basename(file_path),
            file_size=file_size,
            worksheets=worksheet_analyses,
            total_boq_tables=len(boq_tables),
            extraction_summary=extraction_summary,
            processing_time=processing_time
        )
        
        return file_analysis


def run_initial_analysis(file_path: str, gemini_api_key: str = None) -> FileAnalysis:
    """
    Instantiates and runs the BOQ pre-processor on a given file.
    
    Args:
        file_path: The path to the uploaded Excel file.
        gemini_api_key: The API key for Gemini services.
        
    Returns:
        A FileAnalysis object containing the results.
    """
    # FIXED: Pass the API key properly to the extractor
    extractor = AdvancedBOQExtractor(gemini_api_key=gemini_api_key)
    analysis_results = extractor.process_excel_file(file_path)
    return analysis_results