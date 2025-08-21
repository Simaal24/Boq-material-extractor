"""
Stage 3 Extractor Module

This is the most powerful module in the engine. It takes the AI-verified headers from Stage 2 
and processes each individual line item, applying the principle of "context inheritance" 
to classify items with high accuracy and consistency.
"""

import asyncio
import json
import logging
from typing import List, Dict, Tuple

# Imports from our new engine modules
from .data_structures import ExtractedLineItem, VerifiedHeader
from .prompts import format_stage3_prompt
from .gemini_client import generate_content
from .patterns import (
    is_location_only_line,
    LOCATION_MAPPING,
    STRUCTURED_MATERIAL_RULES
)

logger = logging.getLogger(__name__)

class Stage3EnhancedContextInheritance:
    """🔧 FIXED: Enhanced context inheritance with location-only detection and grade preservation"""

    def __init__(self):
        self.api_calls = 0
        logger.info(f"🔧 Stage 3 ENHANCED: Context inheritance with decision tracking")

    def extract_location(self, description: str) -> str:
        """Extract standardized location from description"""
        text_lower = description.lower()
        
        # Check each location category
        for location, keywords in LOCATION_MAPPING.items():
            if any(keyword in text_lower for keyword in keywords):
                return location
        
        # Default fallback
        return "Other Building Elements"

    def extract_structured_material(self, description: str, category: str) -> Tuple[str, str, str]:
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
                material = rules.get("material_output", "Concrete")
            
            # Extract grade
            if "grade_pattern" in rules:
                import re
                grade_match = re.search(rules["grade_pattern"], text_lower)
                if grade_match:
                    grade = rules["grade_format"].format(grade_match.group(1))
        
        elif category == "REINFORCEMENT_STEEL":
            material = rules.get("material_output", "Reinforcement Steel")
            if "grade_pattern" in rules:
                import re
                grade_match = re.search(rules["grade_pattern"], text_lower)
                if grade_match:
                    grade = rules["grade_format"].format(grade_match.group(1))
        
        elif category == "STEEL_WORKS":
            material = rules.get("material_output", "Structural Steel")
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
                import re
                grade_match = re.search(rules["grade_pattern"], text_lower)
                if grade_match:
                    grade = rules["grade_format"].format(grade_match.group(1))
            
            # Extract dimensions
            if category == "MASONRY_WALL" and "dimension_pattern" in rules:
                import re
                dim_match = re.search(rules["dimension_pattern"], text_lower)
                if dim_match:
                    dimensions = f"{dim_match.group(1)}x{dim_match.group(2)}x{dim_match.group(3)} mm"
            elif category == "PLASTERING_WORK" and "thickness_pattern" in rules:
                import re
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
                import re
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

    async def extract_chunk_details_enhanced(self, chunk_info: Dict, verified_header: VerifiedHeader, chunk_items: List[Tuple]) -> List[ExtractedLineItem]:
        """🔧 FIXED: Enhanced context inheritance with location-only detection"""
        if not chunk_items:
            return []

        # Scalable internal batching
        OPTIMAL_BATCH_SIZE = 12
        
        if len(chunk_items) <= OPTIMAL_BATCH_SIZE:
            return await self._process_batch_with_enhanced_inheritance(chunk_info, verified_header, chunk_items, batch_num=1, total_batches=1)
        else:
            logger.info(f"🔧 ENHANCED INHERITANCE: Chunk {chunk_info['chunk_id']} has {len(chunk_items)} items - breaking into internal batches of {OPTIMAL_BATCH_SIZE}")
            
            all_extracted_items = []
            total_batches = (len(chunk_items) + OPTIMAL_BATCH_SIZE - 1) // OPTIMAL_BATCH_SIZE
            
            for i in range(0, len(chunk_items), OPTIMAL_BATCH_SIZE):
                batch = chunk_items[i:i + OPTIMAL_BATCH_SIZE]
                batch_num = (i // OPTIMAL_BATCH_SIZE) + 1
                
                logger.info(f"🔧 Processing enhanced inheritance batch {batch_num}/{total_batches} ({len(batch)} items)")
                
                batch_results = await self._process_batch_with_enhanced_inheritance(
                    chunk_info, verified_header, batch, batch_num, total_batches
                )
                all_extracted_items.extend(batch_results)
            
            logger.info(f"🔧 ENHANCED INHERITANCE: Completed {total_batches} internal batches for chunk {chunk_info['chunk_id']}")
            return all_extracted_items

    async def _process_batch_with_enhanced_inheritance(self, chunk_info: Dict, verified_header: VerifiedHeader, batch_items: List[Tuple], batch_num: int, total_batches: int) -> List[ExtractedLineItem]:
        """🔧 FIXED: Process batch with enhanced context inheritance rules"""
        MAX_RETRIES = 3
        BASE_DELAY = 1.0
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔧 INHERITANCE RETRY: Attempt {attempt + 1}/{MAX_RETRIES + 1} for batch {batch_num}")
                
                return await self._process_single_batch_enhanced(chunk_info, verified_header, batch_items, batch_num, total_batches)
                
            except json.JSONDecodeError as e:
                logger.warning(f"🔧 JSON decode error (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES:
                    logger.error(f"🔧 JSON parsing failed after {MAX_RETRIES + 1} attempts")
                    return self._create_enhanced_inheritance_fallback(chunk_info, verified_header, batch_items, f"JSON decode failed after {MAX_RETRIES + 1} attempts")
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for non-retriable errors
                non_retriable_errors = [
                    "api key not valid", "invalid api key", "unauthorized", 
                    "403", "401", "billing", "quota exceeded permanently",
                    "rate limit exceeded", "resource exhausted"
                ]
                
                is_non_retriable = any(phrase in error_msg for phrase in non_retriable_errors)
                
                if is_non_retriable:
                    logger.error(f"🔧 Non-retriable error: {e}")
                    return self._create_enhanced_inheritance_fallback(chunk_info, verified_header, batch_items, f"Non-retriable error: {str(e)}")
                
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** attempt)
                    logger.warning(f"🔧 Retrying in {delay}s (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"🔧 All retries exhausted: {e}")
                    return self._create_enhanced_inheritance_fallback(chunk_info, verified_header, batch_items, f"API failed after {MAX_RETRIES + 1} attempts: {str(e)}")
        
        return self._create_enhanced_inheritance_fallback(chunk_info, verified_header, batch_items, "Unexpected retry loop exit")

    async def _process_single_batch_enhanced(self, chunk_info: Dict, verified_header: VerifiedHeader, batch_items: List[Tuple], batch_num: int, total_batches: int) -> List[ExtractedLineItem]:
        """🔧 FIXED: Simplified context inheritance with STRICT rules"""
        
        # Pre-process items to detect location-only lines
        processed_items = []
        for i, (row_idx, desc, unit) in enumerate(batch_items):
            is_location_only = is_location_only_line(desc)
            processed_items.append({
                "item_id": i + 1,
                "row_index": row_idx,
                "description": desc,
                "unit": unit or "N/A",
                "is_location_only": is_location_only
            })
        
        # Build simplified prompt focusing on INHERITANCE FIRST
        items_text = "\n".join([
            f"Item {item['item_id']} (Row {item['row_index']}): \"{item['description']}\" | Unit: \"{item['unit']}\" | Location-only: {item['is_location_only']}"
            for item in processed_items
        ])
        
        # Use our centralized prompt formatting
        final_prompt = format_stage3_prompt(
            verified_category=verified_header.verified_category,
            verified_material=verified_header.verified_material,
            header_specifications=str(verified_header.header_specifications),
            items_text=items_text
        )
        
        # Use our new central API client
        extraction_data = await generate_content(final_prompt)
        self.api_calls += 1
        
        if extraction_data is None:
            logger.error(f"AI call failed for line item batch {batch_num}. Falling back.")
            return self._create_enhanced_inheritance_fallback(chunk_info, verified_header, batch_items, "AI call failed")
        
        # Convert AI results to ExtractedLineItem objects
        extracted_items = []
        
        try:
            # Handle both list and dict responses
            if isinstance(extraction_data, list):
                response_list = extraction_data
            elif isinstance(extraction_data, dict) and 'results' in extraction_data:
                response_list = extraction_data['results']
            else:
                # Try to extract from the response
                response_list = [extraction_data] if isinstance(extraction_data, dict) else []
            
            for j, item_data in enumerate(response_list):
                if j < len(batch_items):
                    row_idx, description, unit = batch_items[j]
                    
                    # Enhanced inheritance tracking
                    context_inherited = item_data.get('context_inheritance_applied', True)
                    material_change_detected = item_data.get('material_change_detected', False)
                    context_decision_reason = item_data.get('context_decision_reason', 'INHERITED_FROM_HEADER')
                    
                    # Enhanced grade and specs handling
                    ai_grade = item_data.get('extracted_grade')
                    ai_dimensions = item_data.get('extracted_dimensions')
                    ai_location = item_data.get('extracted_location')
                    ai_tech_specs = item_data.get('technical_specs')
                    
                    # Fallback to Python extraction if AI left fields empty
                    if not ai_grade or not ai_dimensions:
                        fallback_material, fallback_grade, fallback_dimensions = self.extract_structured_material(
                            description, item_data.get('final_category', verified_header.verified_category)
                        )
                        ai_grade = ai_grade or fallback_grade
                        ai_dimensions = ai_dimensions or fallback_dimensions
                    
                    if not ai_location:
                        ai_location = self.extract_location(description)
                    
                    # Preserve header specifications if context inherited
                    final_specs = verified_header.header_specifications.copy() if context_inherited else {}
                    
                    extracted_items.append(ExtractedLineItem(
                        original_row_index=row_idx, 
                        original_description=description, 
                        original_unit=unit,
                        chunk_id=chunk_info['chunk_id'], 
                        final_category=item_data.get('final_category', verified_header.verified_category),
                        final_material=item_data.get('final_material', verified_header.verified_material),
                        extracted_grade=ai_grade, 
                        extracted_dimensions=ai_dimensions,
                        extracted_location=ai_location, 
                        technical_specs=ai_tech_specs,
                        anomaly_flag=item_data.get('anomaly_flag', False),
                        anomaly_reason=item_data.get('material_change_reason') if material_change_detected else None,
                        confidence=item_data.get('confidence', 90.0),
                        extraction_notes=f"🔧 Enhanced Inheritance: {item_data.get('ai_reasoning', 'Context processed')}" + (" (Confidence defaulted)" if 'confidence' not in item_data else ""),
                        processing_stage='STAGE_3_ENHANCED_INHERITANCE', 
                        processing_time=0.1,
                        context_applied=context_inherited,
                        extraction_method="ENHANCED_CONTEXT_INHERITANCE",
                        unit_verified=True,
                        material_change_detected=material_change_detected,
                        specifications_inherited=final_specs,
                        context_decision_reason=context_decision_reason
                    ))
            
            logger.info(f"🔧 ENHANCED INHERITANCE SUCCESS: Batch {batch_num} processed {len(extracted_items)} items with decision tracking")
            return extracted_items
            
        except Exception as e:
            logger.error(f"Error processing AI response for batch {batch_num}: {e}")
            return self._create_enhanced_inheritance_fallback(chunk_info, verified_header, batch_items, f"Response processing error: {str(e)}")

    def _create_enhanced_inheritance_fallback(self, chunk_info: Dict, verified_header: VerifiedHeader, batch_items: List[Tuple], error_reason: str) -> List[ExtractedLineItem]:
        """🔧 FIXED: Strict inheritance fallback - ALWAYS inherit header context"""
        logger.info(f"🔧 STRICT INHERITANCE FALLBACK: All {len(batch_items)} items will inherit header context")
        
        fallback_items = []
        for row_idx, description, unit in batch_items:
            try:
                # ✅ CRITICAL: Always inherit from header in fallback - context never lost
                category = verified_header.verified_category  # Never change this
                material = verified_header.verified_material  # Never change this
                
                # ALWAYS use header specifications - preserve grade
                header_specs = verified_header.header_specifications or {}
                grade = header_specs.get('grade', '')  # Preserve grade
                thickness = header_specs.get('thickness', '')
                ratio = header_specs.get('ratio', '')
                
                # Only extract location - everything else inherited
                location = self.extract_location(description)
                
                # Build technical specs from header specifications
                tech_specs_parts = []
                if grade:
                    tech_specs_parts.append(f"Grade: {grade}")
                if thickness:
                    tech_specs_parts.append(f"Thickness: {thickness}")
                if ratio:
                    tech_specs_parts.append(f"Ratio: {ratio}")
                
                tech_specs = "; ".join(tech_specs_parts) if tech_specs_parts else "Inherited from header"
                
                # Always inherit - no material change detection in fallback
                context_decision_reason = "FALLBACK_BUT_CONTEXT_PRESERVED"
                confidence = 85.0  # High confidence in inheritance
                
                fallback_items.append(ExtractedLineItem(
                    original_row_index=row_idx, 
                    original_description=description, 
                    original_unit=unit,
                    chunk_id=chunk_info['chunk_id'], 
                    final_category=category,
                    final_material=material, 
                    extracted_grade=grade, 
                    extracted_dimensions=thickness,
                    extracted_location=location, 
                    technical_specs=tech_specs, 
                    anomaly_flag=True,
                    anomaly_reason=f"🔧 Strict Inheritance Fallback: {error_reason}", 
                    confidence=confidence,
                    extraction_notes=f'🔧 Strict inheritance fallback - all context inherited from header', 
                    processing_stage='STAGE_3_STRICT_INHERITANCE_FALLBACK', 
                    processing_time=0.001,
                    context_applied=True,  # Always true in fallback
                    extraction_method='STRICT_INHERITANCE_FALLBACK',
                    unit_verified=False, 
                    material_change_detected=False,  # Never in fallback
                    specifications_inherited=verified_header.header_specifications,
                    context_decision_reason=context_decision_reason
                ))
                
            except Exception as fallback_error:
                # Ultimate fallback - still inherit header context
                fallback_items.append(ExtractedLineItem(
                    original_row_index=row_idx, 
                    original_description=description, 
                    original_unit=unit,
                    chunk_id=chunk_info['chunk_id'], 
                    final_category=verified_header.verified_category,
                    final_material=verified_header.verified_material, 
                    extracted_grade=verified_header.header_specifications.get('grade', ''), 
                    extracted_dimensions=verified_header.header_specifications.get('thickness', ''),
                    extracted_location="Other Building Elements", 
                    technical_specs="Ultimate inheritance fallback", 
                    anomaly_flag=True,
                    anomaly_reason=f"🔧 Ultimate Inheritance Fallback: {error_reason} + {fallback_error}",
                    confidence=80.0, 
                    extraction_notes='🔧 Ultimate inheritance fallback - header context preserved',
                    processing_stage='STAGE_3_ULTIMATE_INHERITANCE_FALLBACK', 
                    processing_time=0.001,
                    context_applied=True,  # Always inherit
                    extraction_method='ULTIMATE_INHERITANCE_FALLBACK',
                    unit_verified=False, 
                    material_change_detected=False,
                    specifications_inherited=verified_header.header_specifications,
                    context_decision_reason='ULTIMATE_INHERITANCE_FALLBACK'
                ))
        
        logger.info(f"🔧 STRICT INHERITANCE FALLBACK: Created {len(fallback_items)} items - ALL inherit header context")
        return fallback_items


def run_stage3_extraction(chunk_info: Dict, verified_header: VerifiedHeader, chunk_items: List[Tuple]) -> List[ExtractedLineItem]:
    """
    Convenience function to run Stage 3 context inheritance extraction.
    
    Args:
        chunk_info: Dictionary containing chunk metadata
        verified_header: VerifiedHeader object from Stage 2
        chunk_items: List of (row_idx, description, unit) tuples
        
    Returns:
        List of ExtractedLineItem objects with context inheritance applied
    """
    async def _run_extraction():
        extractor = Stage3EnhancedContextInheritance()
        return await extractor.extract_chunk_details_enhanced(chunk_info, verified_header, chunk_items)
    
    return asyncio.run(_run_extraction())