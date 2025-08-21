"""
BOQ Data Structures Module

This module contains all the data structure blueprints (dataclasses) used throughout
the BOQ processing pipeline. These ensure consistent data shapes across all stages.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

@dataclass
class ProcessingChunk:
    """Represents a chunk of BOQ items to be processed together"""
    chunk_id: int
    header_info: Dict
    items: List[Tuple[int, str, str]]  # (row_index, description, unit)
    context_hint: str
    estimated_complexity: int
    header_specifications: Dict = field(default_factory=dict)

@dataclass
class RuleResult:
    """Result from rule-based classification"""
    success: bool
    category: str = "OTHERS"
    material: str = "Mixed"
    confidence: float = 0.0
    method: str = "RULE"
    context_applied: bool = False
    unit_verified: bool = False
    material_change_detected: bool = False
    specifications_inherited: Dict = field(default_factory=dict)
    context_decision_reason: str = ""  # Track decision reasoning

@dataclass
class VerifiedHeader:
    """Header that has been verified and corrected by AI"""
    chunk_id: int
    original_row_index: int
    header_text: str
    verified_category: str
    verified_material: str
    verification_status: str
    correction_reason: str
    confidence: float
    primary_activity: str
    material_grade: Optional[str]
    ambiguity_resolved: Optional[str]
    header_specifications: Dict = field(default_factory=dict)

@dataclass
class ExtractedLineItem:
    """Complete extracted and enriched line item"""
    original_row_index: int
    original_description: str
    original_unit: str
    chunk_id: int
    final_category: str
    final_material: str
    extracted_grade: Optional[str]
    extracted_dimensions: Optional[str]
    extracted_location: Optional[str]
    technical_specs: Optional[str]
    anomaly_flag: bool
    anomaly_reason: Optional[str]
    confidence: float
    extraction_notes: str
    processing_stage: str
    processing_time: float
    context_applied: bool = False
    extraction_method: str = "UNKNOWN"
    unit_verified: bool = False
    material_change_detected: bool = False
    specifications_inherited: Dict = field(default_factory=dict)
    context_decision_reason: str = ""  # Track decision reasoning

@dataclass
class WorksheetExtraction:
    """Results from processing a single worksheet"""
    worksheet_name: str
    total_rows: int
    headers_detected: int
    headers_verified: int
    items_extracted: int
    processing_time: float
    stage_1_time: float
    stage_2_time: float
    stage_3_time: float
    line_items: List[ExtractedLineItem] = field(default_factory=list)
    verified_headers: List[VerifiedHeader] = field(default_factory=list)

@dataclass
class FileExtraction:
    """Complete results from processing an entire file"""
    original_filename: str
    worksheets: List[WorksheetExtraction]
    total_api_calls: int
    total_processing_time: float
    summary: Dict