"""
Stage 2 Verifier Module

This module acts as the first AI quality check. It takes the header guesses from Stage 1 
and uses sophisticated prompts to ask Gemini AI to either confirm or correct them.
This ensures high accuracy before moving on to the line items.
"""

import asyncio
import json
import logging
from typing import List, Dict

# Imports from our new engine modules
from .data_structures import VerifiedHeader
from .prompts import format_stage2_prompt
from .gemini_client import generate_content

logger = logging.getLogger(__name__)

class Stage2EnhancedHeaderVerifier:
    """🔧 AI header verification with explicit activity priority examples"""

    def __init__(self, batch_size: int = 5):
        self.api_calls = 0
        self.batch_size = batch_size
        logger.info(f"🔧 Stage 2 Enhanced Header Verifier initialized with batch size: {batch_size}")

    async def _process_single_header_batch(self, batch_num: int, batch: List[Dict], total_batches: int) -> List[VerifiedHeader]:
        """Process ONE batch of headers with parallel execution"""
        headers_text = "\n".join([
            f'Header {j} (Chunk {h["chunk_id"]}, Row {h["header_row"]}): "{h["header_text"]}" | Stage1: {h["rule_category"]} | {h["rule_material"]} | Specs: {h.get("header_specifications", {})}'
            for j, h in enumerate(batch, 1)
        ])
        
        # Use our centralized prompt formatting
        final_prompt = format_stage2_prompt(headers_text)
        
        logger.info(f"🚀 Processing header batch {batch_num}/{total_batches} with activity priority")
        
        # Use our new central API client to get the AI response
        # The client already handles retries and errors
        verified_data = await generate_content(final_prompt)
        self.api_calls += 1
        
        if verified_data is None:
            logger.error(f"AI call failed for header batch {batch_num}. Falling back.")
            return self._create_fallback_results(batch)
        
        # Process the successful response
        batch_results = []
        
        try:
            # Handle both list and dict responses
            if isinstance(verified_data, list):
                response_list = verified_data
            elif isinstance(verified_data, dict) and 'results' in verified_data:
                response_list = verified_data['results']
            else:
                # Try to extract from the response
                response_list = [verified_data] if isinstance(verified_data, dict) else []
            
            for item in response_list:
                # Find corresponding header info
                header_info = next((h for h in batch if h['chunk_id'] == item.get('chunk_id')), None)
                
                batch_results.append(VerifiedHeader(
                    chunk_id=item.get('chunk_id', 0),
                    original_row_index=item.get('original_row_index', header_info['header_row'] if header_info else 0),
                    header_text=header_info['header_text'] if header_info else '',
                    verified_category=item.get('verified_category', 'OTHERS'),
                    verified_material=item.get('verified_material', 'Mixed Materials'),
                    verification_status=item.get('verification_status', 'CONFIRMED'),
                    correction_reason=item.get('correction_reason', ''),
                    confidence=item.get('confidence', 85.0),
                    primary_activity=item.get('primary_activity', 'Unknown'),
                    material_grade=item.get('material_grade'),
                    ambiguity_resolved=item.get('ambiguity_resolved'),
                    header_specifications=item.get('header_specifications', {})
                ))
                
                # Log ALL verifications and corrections
                if item.get('verification_status') == 'CORRECTED':
                    logger.info(f"🔧 ACTIVITY CORRECTED Chunk {item.get('chunk_id')} (Row {item.get('original_row_index')}): {header_info['rule_category'] if header_info else 'Unknown'} → {item.get('verified_category')}")
                else:
                    logger.info(f"✅ ACTIVITY CONFIRMED Chunk {item.get('chunk_id')} (Row {item.get('original_row_index')}): {item.get('verified_category')}")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error processing AI response for batch {batch_num}: {e}")
            return self._create_fallback_results(batch)

    def _create_fallback_results(self, batch: List[Dict]) -> List[VerifiedHeader]:
        """Create fallback VerifiedHeader objects when AI fails"""
        fallback_results = []
        for header in batch:
            fallback_results.append(VerifiedHeader(
                chunk_id=header['chunk_id'], 
                original_row_index=header.get('header_row', 0),
                header_text=header['header_text'],
                verified_category=header['rule_category'],
                verified_material=header['rule_material'], 
                verification_status='FALLBACK',
                correction_reason=f'AI failed, using Stage 1 results', 
                confidence=50.0,
                primary_activity='Unknown', 
                material_grade=None, 
                ambiguity_resolved=None,
                header_specifications=header.get('header_specifications', {})
            ))
        
        logger.info(f"🔧 Created {len(fallback_results)} fallback verified headers")
        return fallback_results

    async def verify_headers(self, header_registry: Dict[int, Dict]) -> List[VerifiedHeader]:
        """🔧 PARALLEL: Header verification with activity priority prompt"""
        if not header_registry:
            logger.warning("🧠 No headers found in registry for verification")
            return []

        # Convert header registry to verification list
        headers_for_verification = []
        for row_idx, header_info in header_registry.items():
            headers_for_verification.append({
                'header_text': header_info['header_text'],
                'rule_category': header_info['stage1_category'],
                'rule_material': header_info['stage1_material'],
                'confidence': 85.0, 
                'chunk_id': header_info['chunk_id'],
                'header_row': header_info['row_index'],
                'header_specifications': header_info.get('header_specifications', {})
            })

        # Create batches
        all_batches = []
        for i in range(0, len(headers_for_verification), self.batch_size):
            batch = headers_for_verification[i:i + self.batch_size]
            all_batches.append(batch)
        
        total_batches = len(all_batches)
        logger.info(f"🚀 Stage 2 PARALLEL: Processing {total_batches} header batches concurrently")
        
        # Create all tasks for parallel execution
        tasks = [
            self._process_single_header_batch(batch_num + 1, batch, total_batches)
            for batch_num, batch in enumerate(all_batches)
        ]
        
        # Execute all tasks in parallel
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_verified_headers = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"⚠ Header batch {i + 1} failed: {result}")
                # Fallback for failed batch
                batch = all_batches[i]
                fallback_headers = self._create_fallback_results(batch)
                all_verified_headers.extend(fallback_headers)
            else:
                all_verified_headers.extend(result)
        
        logger.info(f"✅ Stage 2 PARALLEL: {len(all_verified_headers)} headers verified concurrently")
        return all_verified_headers


def run_stage2_verification(header_registry: Dict[int, Dict], batch_size: int = 5) -> List[VerifiedHeader]:
    """
    Convenience function to run Stage 2 header verification.
    
    Args:
        header_registry: Dictionary of header information from Stage 1
        batch_size: Number of headers to process in each batch
        
    Returns:
        List of VerifiedHeader objects with AI corrections/confirmations
    """
    async def _run_verification():
        verifier = Stage2EnhancedHeaderVerifier(batch_size=batch_size)
        return await verifier.verify_headers(header_registry)
    
    return asyncio.run(_run_verification())