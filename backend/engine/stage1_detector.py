"""
Stage 1 Detector Module

This module contains the initial, rule-based analyzer that reads raw data and makes
fast, educated guesses about document structure without using AI. It identifies
likely headers and groups line items into chunks for further AI processing.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Imports from our new engine modules
from .data_structures import ProcessingChunk
from .patterns import (
    PRIMARY_VERBS,
    MATERIAL_INDICATORS,
    COMPREHENSIVE_INDICATORS,
    TECHNICAL_INDICATORS,
    REJECT_PATTERNS,
    NOTE_PATTERNS,
    STRUCTURAL_EXCLUSIONS,
    MATERIAL_CATEGORY_MAP,
    LOCATION_MAPPING,
    STRUCTURED_MATERIAL_RULES,
    CONCRETE_GRADE_PATTERNS,
    MORTAR_PATTERNS,
    STEEL_GRADE_PATTERNS,
    SPECIFIC_MATERIALS,
    get_activity_priority_category,
    is_complex_header,
    is_incidental_material_mention,
    extract_header_specifications
)

logger = logging.getLogger(__name__)

class Stage1EnhancedHeaderDetector:
    """ENHANCED header detection with complex patterns and specification extraction"""

    def __init__(self):
        logger.info("⚡ Stage 1 ENHANCED: Complex Header Detection with Specification Extraction + Activity Priority")

    def analyze_boq_structure(self, text: str) -> Dict:
        """ENHANCED BOQ structure analysis with better patterns"""
        text_lower = text.lower().strip()
        text_lower = re.sub(r'^"+|"+$', '', text_lower)
        
        analysis = {
            "primary_verb": None, "secondary_action": None, "material_type": None, "material_grade": None,
            "is_activity": False, "confidence": 0.0, "has_comprehensive_scope": False,
            "length_score": 0.0, "technical_score": 0.0, "scope_count": 0, "tech_count": 0,
            "is_complex_header": False
        }
        
        # NEW: Check for complex header patterns first
        analysis["is_complex_header"] = is_complex_header(text)
        if analysis["is_complex_header"]:
            analysis["confidence"] += 25.0
            logger.debug(f"🔍 Complex header detected: {text[:50]}...")
        
        # Step 1: Primary Verb Detection
        for verb_category, verb_data in PRIMARY_VERBS.items():
            for variant in verb_data["variants"]:
                if variant in text_lower[:50]:
                    analysis["primary_verb"] = verb_category
                    analysis["is_activity"] = True
                    analysis["confidence"] += 30.0
                    break
            if analysis["primary_verb"]:
                break
        
        # Step 2: Secondary Action Detection
        if analysis["primary_verb"]:
            verb_data = PRIMARY_VERBS[analysis["primary_verb"]]
            for action in verb_data["secondary_actions"].keys():
                if action in text_lower:
                    analysis["secondary_action"] = action
                    analysis["confidence"] += 20.0
                    break
        
        # Step 3: Enhanced Material Detection
        best_material_score = 0
        best_material_type = None
        
        for material_category, material_data in MATERIAL_INDICATORS.items():
            material_score = 0
            material_score += sum(15.0 for indicator in material_data["primary"] if indicator in text_lower)
            material_score += sum(5.0 for indicator in material_data["secondary"] if indicator in text_lower)
            
            if "exclusions" in material_data:
                material_score -= sum(10.0 for exclusion in material_data["exclusions"] if exclusion in text_lower)
            
            # Grade patterns
            if material_score > 0:
                for pattern_key in ["ratio_patterns", "grade_patterns"]:
                    if pattern_key in material_data:
                        for pattern in material_data[pattern_key]:
                            if re.search(pattern, text_lower):
                                analysis["material_grade"] = re.search(pattern, text_lower).group(1)
                                material_score += 10.0
                                break
            
            if material_score > best_material_score:
                best_material_score = material_score
                best_material_type = material_category
        
        if best_material_type:
            analysis["material_type"] = best_material_type
            analysis["confidence"] += best_material_score
        
        # Step 4: Scope indicators
        analysis["scope_count"] = sum(1 for indicator in COMPREHENSIVE_INDICATORS if indicator in text_lower)
        if analysis["scope_count"] >= 1:
            analysis["has_comprehensive_scope"] = True
            analysis["confidence"] += analysis["scope_count"] * 3.0
        
        # Step 5: Length scoring
        text_length = len(text_lower)
        if text_length > 200: analysis["length_score"] = 20.0
        elif text_length > 150: analysis["length_score"] = 15.0
        elif text_length > 100: analysis["length_score"] = 10.0
        elif text_length > 60: analysis["length_score"] = 5.0
        analysis["confidence"] += analysis["length_score"]
        
        # Step 6: Technical indicators
        analysis["tech_count"] = sum(1 for indicator in TECHNICAL_INDICATORS if indicator in text_lower)
        analysis["technical_score"] = analysis["tech_count"] * 2.0
        analysis["confidence"] += analysis["technical_score"]
        
        return analysis

    def is_main_header(self, description: str) -> bool:
        """ENHANCED 4-tier header detection with complex patterns"""
        text = (description or "").strip()
        if not text or len(text) < 30:
            return False

        text_lower = text.lower()
        
        # Rejection patterns
        if any(re.search(pattern, text_lower) for pattern in REJECT_PATTERNS):
            return False

        analysis = self.analyze_boq_structure(text)
        
        # 4-TIER MATCHING SYSTEM
        is_complex_match = (
            analysis["is_complex_header"] and analysis["is_activity"] and 
            analysis["primary_verb"] and len(text_lower) >= 80 and 
            analysis["tech_count"] >= 2
        )
        
        is_perfect_match = (
            analysis["is_activity"] and analysis["primary_verb"] and analysis["secondary_action"] and 
            analysis["material_type"] and analysis["has_comprehensive_scope"] and 
            len(text_lower) >= 60 and analysis["tech_count"] >= 2
        )
        
        is_structural_match = (
            analysis["is_activity"] and analysis["primary_verb"] and analysis["secondary_action"] and 
            not analysis["material_type"] and analysis["has_comprehensive_scope"] and 
            len(text_lower) >= 60 and analysis["tech_count"] >= 2 and analysis["confidence"] > 75
        )
        
        is_short_header_match = (
            analysis["is_activity"] and analysis["primary_verb"] and analysis["secondary_action"] and 
            analysis["material_type"] and len(text_lower) < 150 and analysis["tech_count"] >= 1
        )
        
        is_header = is_complex_match or is_perfect_match or is_structural_match or is_short_header_match
        
        if is_header:
            if is_complex_match:
                match_type = "Complex"
            elif is_perfect_match:
                match_type = "Perfect"
            elif is_structural_match:
                match_type = "Structural"
            else:
                match_type = "Short"
            logger.debug(f"🏆 HEADER DETECTED ({match_type}): {analysis.get('primary_verb')} + {analysis.get('secondary_action')}")
        
        return is_header

    def infer_category_from_header(self, text: str) -> str:
        """🔧 UPDATED: Activity priority over material keywords"""
        analysis = self.analyze_boq_structure(text)
        text_lower = text.lower()
        
        # NEW: Priority 2: Activity-based classification
        activity_category, _, confidence = get_activity_priority_category(text)
        if activity_category and confidence >= 85.0:
            logger.info(f"🎯 ACTIVITY PRIORITY: {activity_category} (confidence: {confidence})")
            return activity_category
        
        # PRIORITY 3: Work type over material type 
        work_type_patterns = [
            (["concrete", "laying concrete", "placing concrete", "rcc", "pcc", "screed concrete"], "STRUCTURE_CONCRETE"),
            (["excavation", "excavating", "digging", "earth work", "soil removal"], "EXCAVATION_EARTHWORK"),
            (["fabricating", "cutting steel", "bending steel", "tmt bars", "steel bars", "reinforcement"], "REINFORCEMENT_STEEL"),
            (["masonry", "brick work", "block work", "wall construction"], "MASONRY_WALL"),
            (["plastering", "rendering", "cement plaster"], "PLASTERING_WORK"),
            (["paint", "painting", "emulsion", "primer"], "PAINTING_FINISHING"),
            (["false ceiling", "ceiling system", "gypsum board"], "OTHERS"),
            (["formwork", "shuttering", "centering"], "FORMWORK_SHUTTERING")
        ]
        
        for keywords, category in work_type_patterns:
            if any(keyword in text_lower for keyword in keywords):
                logger.info(f"🎯 Work type priority: {category} for keywords: {keywords}")
                return category
        
        # PRIORITY 4: Direct keyword matching for specific categories
        direct_category_matches = [
            (["surface excavation", "excavation exceeding", "excavation in", "earthwork"], "EXCAVATION_EARTHWORK"),
            (["concrete laying", "providing and laying concrete", "pcc", "rcc", "concrete slab"], "STRUCTURE_CONCRETE"),
            (["reinforcement steel", "tmt bars", "steel bars", "deformed bars"], "REINFORCEMENT_STEEL"),
            (["brick work", "masonry work", "aac block", "solid block"], "MASONRY_WALL"),
            (["formwork", "shuttering", "centering"], "FORMWORK_SHUTTERING"),
            (["plaster", "plastering"], "PLASTERING_WORK"),
            (["flooring", "granite slab", "tiles", "vitrified"], "FLOORING_TILE"),  # Updated: Only granite slab
            (["waterproofing", "damp proof"], "WATERPROOFING_MEMBRANE"),
            (["paint", "painting", "emulsion", "primer"], "PAINTING_FINISHING"),
            (["door", "window", "frame"], "JOINERY_DOORS"),
            (["steel work", "structural steel"], "STEEL_WORKS"),
            (["electrical", "wiring", "conduit"], "MEP_ELECTRICAL"),
            (["plumbing", "pipe", "fitting"], "MEP_PLUMBING")
        ]
        
        for keywords, category in direct_category_matches:
            if any(keyword in text_lower for keyword in keywords):
                # NEW: Check if granite is incidental
                if "granite" in text_lower and is_incidental_material_mention(text, "granite"):
                    continue  # Skip granite aggregate mentions
                logger.info(f"🎯 Direct category match: {category} for keywords: {keywords}")
                return category
        
        # PRIORITY 5: Verb-based overrides
        if analysis.get("primary_verb") == "EXCAVATION": 
            return "EXCAVATION_EARTHWORK"
        elif analysis.get("primary_verb") in ["PAINTING_VERB", "PREPARE_SURFACE"]: 
            return "PAINTING_FINISHING"
        elif analysis.get("primary_verb") == "DEMOLITION": 
            return "DEMOLITION"
        
        # PRIORITY 6: Material-based classification (only if not incidental)
        if analysis.get("material_type") and analysis["material_type"] in MATERIAL_CATEGORY_MAP:
            return MATERIAL_CATEGORY_MAP[analysis["material_type"]]
        
        logger.warning(f"⚠️ Category fallback to OTHERS for: {text_lower[:50]}...")
        return "OTHERS"

    def infer_material_from_header(self, text: str) -> str:
        """🔧 UPDATED: Material inference with demolition and activity priority"""
        text_lower = text.lower()
        
        # PRIORITY 2: Specialized materials
        from .patterns import check_specialized_material
        specialized_category, specialized_material = check_specialized_material(text)
        if specialized_material:
            logger.info(f"🎯 Specialized material: {specialized_material}")
            return specialized_material
        
        # PRIORITY 3: Activity-based materials
        activity_category, activity_material, confidence = get_activity_priority_category(text)
        if activity_category and confidence >= 85.0:
            if activity_material:
                return activity_material
        
        # PRIORITY 4: Specific materials
        for keywords, material in SPECIFIC_MATERIALS:
            if any(keyword in text_lower for keyword in keywords):
                return material
        
        # PRIORITY 5: Mortar patterns
        for pattern, template in MORTAR_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return template.format(match.group(1).replace(" ", ""))
        
        # PRIORITY 6: Concrete grades
        for pattern, template in CONCRETE_GRADE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    grade_num = float(match.group(1))
                    if 5 <= grade_num <= 80:
                        return template.format(match.group(1))
                except ValueError:
                    continue
        
        # PRIORITY 7: Steel grades
        for pattern, template in STEEL_GRADE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match and match.group(1) in ["415", "500", "550", "250"]:
                return template.format(match.group(1))
        
        # PRIORITY 8: Enhanced general materials (avoid incidental mentions)
        enhanced_materials = [
            (["surface excavation", "excavation exceeding", "excavated earth"], "Excavated Soil"),
            (["concrete", "pcc", "rcc"], "Concrete"),
            (["reinforcement", "steel", "tmt", "bars"], "Steel"),
            (["brick", "masonry", "block"], "Masonry"),
            (["granite slab", "granite flooring", "granite tile"], "Granite"),  # Only specific granite uses
            (["tiles", "flooring", "vitrified"], "Flooring Material"),
            (["plaster", "plastering"], "Cement Plaster"),
            (["paint", "emulsion", "primer", "painting"], "Paint"),
            (["waterproofing", "damp proof"], "Waterproofing Material"),
            (["polythene", "plastic sheet"], "Polythene Sheet"),
            (["formwork", "shuttering"], "Formwork Material")
        ]
        
        for keywords, material in enhanced_materials:
            if any(keyword in text_lower for keyword in keywords):
                return material
        
        return "Mixed Materials"

    def extract_structured_material(self, description: str, category: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Extract structured material, grade, and dimensions based on category"""
        text_lower = description.lower()
        
        if category not in STRUCTURED_MATERIAL_RULES:
            return "Mixed Materials", None, None
        
        rules = STRUCTURED_MATERIAL_RULES[category]
        material = None
        grade = None
        dimensions = None
        
        # Extract based on category-specific rules
        if category == "STRUCTURE_CONCRETE":
            # Check for special types first
            for special_key, special_material in rules.get("special_types", {}).items():
                if special_key in text_lower:
                    material = special_material
                    break
            
            if not material:
                material = rules["material_output"]
            
            # Extract grade
            if "grade_pattern" in rules:
                grade_match = re.search(rules["grade_pattern"], text_lower)
                if grade_match:
                    grade = rules["grade_format"].format(grade_match.group(1))
        
        elif category == "REINFORCEMENT_STEEL":
            material = rules["material_output"]
            if "grade_pattern" in rules:
                grade_match = re.search(rules["grade_pattern"], text_lower)
                if grade_match:
                    grade = rules["grade_format"].format(grade_match.group(1))
        
        elif category == "STEEL_WORKS":
            material = rules["material_output"]
            if "grade_format" in rules:
                grade = rules["grade_format"]
        
        elif category in ["MASONRY_WALL", "PLASTERING_WORK"]:
            # Find specific material type
            for material_type in rules.get("material_types", []):
                if any(word in text_lower for word in material_type.lower().split()):
                    material = material_type
                    break
            
            if not material:
                material = rules["material_types"][0] if rules.get("material_types") else "Mixed"
            
            # Extract grade (ratio)
            if "grade_pattern" in rules:
                grade_match = re.search(rules["grade_pattern"], text_lower)
                if grade_match:
                    grade = rules["grade_format"].format(grade_match.group(1))
            
            # Extract dimensions
            if category == "MASONRY_WALL" and "dimension_pattern" in rules:
                dim_match = re.search(rules["dimension_pattern"], text_lower)
                if dim_match:
                    dimensions = f"{dim_match.group(1)}x{dim_match.group(2)}x{dim_match.group(3)} mm"
            elif category == "PLASTERING_WORK" and "thickness_pattern" in rules:
                thick_match = re.search(rules["thickness_pattern"], text_lower)
                if thick_match:
                    dimensions = f"{thick_match.group(1)}mm thickness"
        
        elif category == "FLOORING_TILE":
            # Find specific material type
            for material_type in rules.get("material_types", []):
                if any(word in text_lower for word in material_type.lower().split()):
                    material = material_type
                    break
            
            if not material:
                material = "Flooring Material"
            
            # Extract thickness
            if "thickness_pattern" in rules:
                thick_match = re.search(rules["thickness_pattern"], text_lower)
                if thick_match:
                    dimensions = f"{thick_match.group(1)}mm thickness"
        
        elif category == "PAINTING_FINISHING":
            # Find specific material type
            for material_type in rules.get("material_types", []):
                if any(word in text_lower for word in material_type.lower().split()):
                    material = material_type
                    break
            
            if not material:
                material = "Paint"
        
        return material or "Mixed Materials", grade, dimensions

    def extract_location(self, description: str) -> str:
        """Extract standardized location from description"""
        text_lower = description.lower()
        
        # Check each location category
        for location, keywords in LOCATION_MAPPING.items():
            if any(keyword in text_lower for keyword in keywords):
                return location
        
        # Default fallback
        return "Other Building Elements"

    def is_empty_or_trivial(self, description: str) -> bool:
        """Check if row is empty"""
        if not description:
            return True
        desc = str(description).strip()
        return desc == '' or desc.lower() in ['nan', 'null', 'none'] or len(desc) < 2

    def is_general_note(self, description: str) -> bool:
        """CURATED note detection with header priority"""
        if not description:
            return False
            
        text = str(description).lower().strip()
        
        # Structural exclusions
        if any(re.search(pattern, text) for pattern in STRUCTURAL_EXCLUSIONS):
            return False
        
        has_note_pattern = any(re.search(pattern, text) for pattern in NOTE_PATTERNS)
        
        # Header takes priority over note
        if has_note_pattern and self.is_main_header(description):
            return False
        
        return has_note_pattern

    def find_description_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find description column"""
        possible_cols = ['description', 'item description', 'work description', 'particulars']
        
        for col in df.columns:
            if any(desc_name in str(col).lower() for desc_name in possible_cols):
                return col
        
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            text_lengths = {col: df[col].astype(str).str.len().mean() for col in text_columns}
            return max(text_lengths, key=text_lengths.get)
        
        return None

    def find_unit_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find unit column in DataFrame"""
        possible_unit_cols = ['unit', 'uom', 'units', 'unit of measurement', 'u.o.m', 'measure']
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(unit_name in col_lower for unit_name in possible_unit_cols):
                logger.info(f"🎯 UNIT COLUMN FOUND: {col}")
                return col
        
        # Check for very short text columns that might be units
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                unique_values = df[col].nunique()
                if avg_length < 10 and unique_values < len(df) * 0.1:
                    sample_values = df[col].dropna().astype(str).str.lower().unique()[:5]
                    unit_indicators = ['cum', 'sqm', 'mt', 'kg', 'nos', 'rmt', 'ltr', 'm', 'each']
                    if any(any(indicator in val for indicator in unit_indicators) for val in sample_values):
                        logger.info(f"🎯 UNIT COLUMN DETECTED (by pattern): {col}")
                        return col
        
        logger.warning("⚠️ No unit column found - proceeding with description-only analysis")
        return None

    def create_sequential_chunks_enhanced(self, df: pd.DataFrame, description_col: str, unit_col: Optional[str] = None) -> Tuple[List[ProcessingChunk], Dict[int, Dict]]:
        """ENHANCED: Create chunks with specification extraction from headers"""
        chunks = []
        header_registry = {}
        current_chunk_items = []
        current_context = {"category": "OTHERS", "material": "Mixed", "header_text": "General Work"}
        current_specifications = {}
        chunk_id = 0
        
        logger.info(f"⚡ Stage 1 ENHANCED: Processing {len(df)} rows with activity priority...")
        if unit_col:
            logger.info(f"🎯 Unit column detected: {unit_col} (used for verification only)")
        else:
            logger.info("⚠️ No unit column found")
        
        for idx, row in df.iterrows():
            description = str(row[description_col]).strip()
            unit = str(row[unit_col]).strip() if unit_col and not pd.isna(row[unit_col]) else ""
            
            if self.is_empty_or_trivial(description):
                continue
            
            if self.is_main_header(description):
                # Save previous chunk ONLY if it has items
                if current_chunk_items:
                    chunks.append(ProcessingChunk(
                        chunk_id=chunk_id,
                        header_info=current_context.copy(),
                        items=current_chunk_items.copy(),
                        context_hint=f"Section: {current_context['category']}",
                        estimated_complexity=len(current_chunk_items),
                        header_specifications=current_specifications.copy()
                    ))
                    chunk_id += 1
                    current_chunk_items = []
                
                # Update context with ENHANCED inference
                inferred_category = self.infer_category_from_header(description)
                inferred_material = self.infer_material_from_header(description)
                
                # Extract specifications from header
                current_specifications = extract_header_specifications(description)
                
                current_context = {
                    "category": inferred_category,
                    "material": inferred_material,
                    "header_text": description,
                    "header_row": idx
                }
                
                # CRITICAL: Register this header for Stage 2 verification
                header_registry[idx] = {
                    "chunk_id": chunk_id,
                    "header_text": description,
                    "stage1_category": inferred_category,
                    "stage1_material": inferred_material,
                    "row_index": idx,
                    "header_specifications": current_specifications.copy()
                }
                
                logger.info(f"📋 HEADER REGISTERED at row {idx} → Chunk {chunk_id}: {inferred_category} | {inferred_material}")
                if current_specifications:
                    logger.info(f"🔍 Header specifications: {current_specifications}")
            
            # Store (row_idx, description, unit) triplets
            current_chunk_items.append((idx, description, unit))
            
            # Smart chunking WITH context preservation
            if len(current_chunk_items) >= 20:
                chunks.append(ProcessingChunk(
                    chunk_id=chunk_id,
                    header_info=current_context.copy(),
                    items=current_chunk_items.copy(),
                    context_hint=f"Section: {current_context['category']} (Part {len(chunks)+1})",
                    estimated_complexity=len(current_chunk_items),
                    header_specifications=current_specifications.copy()
                ))
                chunk_id += 1
                current_chunk_items = []
        
        # Final chunk
        if current_chunk_items:
            chunks.append(ProcessingChunk(
                chunk_id=chunk_id,
                header_info=current_context.copy(),
                items=current_chunk_items.copy(),
                context_hint=f"Section: {current_context['category']} (Final)",
                estimated_complexity=len(current_chunk_items),
                header_specifications=current_specifications.copy()
            ))
        
        logger.info(f"✅ Stage 1 Complete: {len(chunks)} chunks created with activity priority, {len(header_registry)} headers registered")
        return chunks, header_registry


def run_stage1_detection(df: pd.DataFrame, description_col: str, unit_col: Optional[str] = None) -> Tuple[List[ProcessingChunk], Dict[int, Dict]]:
    """
    Convenience function to run Stage 1 detection on a DataFrame.
    
    Args:
        df: The DataFrame containing BOQ data
        description_col: Name of the column containing descriptions
        unit_col: Name of the column containing units (optional)
        
    Returns:
        Tuple of (chunks, header_registry) from Stage 1 processing
    """
    detector = Stage1EnhancedHeaderDetector()
    return detector.create_sequential_chunks_enhanced(df, description_col, unit_col)

def find_description_column(df: pd.DataFrame) -> Optional[str]:
    """
    Convenience function to find the description column in a DataFrame.
    
    Args:
        df: The DataFrame to analyze
        
    Returns:
        Name of the description column or None if not found
    """
    detector = Stage1EnhancedHeaderDetector()
    return detector.find_description_column(df)

def find_unit_column(df: pd.DataFrame) -> Optional[str]:
    """
    Convenience function to find the unit column in a DataFrame.
    
    Args:
        df: The DataFrame to analyze
        
    Returns:
        Name of the unit column or None if not found
    """
    detector = Stage1EnhancedHeaderDetector()
    return detector.find_unit_column(df)