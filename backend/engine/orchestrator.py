"""
BOQ Orchestrator Module

This module contains the main pipeline logic that coordinates all stages of BOQ processing.
It runs Stage 1 (detection), Stage 2 (verification), and Stage 3 (extraction) in sequence,
then provides functions for final summarization.
"""

import pandas as pd
import asyncio
import logging
from typing import List, Dict
from io import BytesIO

# Import all the engine parts we have built
from .stage1_detector import Stage1EnhancedHeaderDetector
from .stage2_verifier import Stage2EnhancedHeaderVerifier
from .stage3_extractor import Stage3EnhancedContextInheritance
from .summarizer import generate_summary_dataframe
from .data_structures import ExtractedLineItem, VerifiedHeader

logger = logging.getLogger(__name__)

async def run_extraction_pipeline(file_path: str, selected_sheets: List[str], api_key: str) -> Dict:
    """
    Orchestrates the entire 3-stage BOQ extraction process for selected sheets.
    
    Args:
        file_path: Path to the Excel file
        selected_sheets: List of worksheet names to process
        api_key: Gemini API key (passed to ensure client is initialized)
        
    Returns:
        Dictionary with results for each sheet
    """
    logger.info(f"🚀 ORCHESTRATOR: Starting 3-stage extraction for {len(selected_sheets)} sheets from {file_path}")
    
    # Initialize all processing stages
    stage1 = Stage1EnhancedHeaderDetector()
    stage2 = Stage2EnhancedHeaderVerifier()  # Uses centralized Gemini client
    stage3 = Stage3EnhancedContextInheritance()  # Uses centralized Gemini client
    
    all_results = {}

    try:
        xls = pd.ExcelFile(file_path)
        
        for sheet_name in selected_sheets:
            logger.info(f"🔄 ORCHESTRATOR: Processing sheet: {sheet_name}")
            
            if sheet_name not in xls.sheet_names:
                logger.warning(f"❌ Sheet {sheet_name} not found in file")
                all_results[sheet_name] = {"error": f"Sheet {sheet_name} not found in file"}
                continue
                
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                logger.warning(f"❌ Sheet {sheet_name} is empty")
                all_results[sheet_name] = {"error": f"Sheet {sheet_name} is empty"}
                continue

            # STAGE 1: Rule-based header detection and chunking
            logger.info(f"📊 STAGE 1: Starting rule-based header detection for {sheet_name}")
            desc_col = stage1.find_description_column(df)
            if not desc_col:
                logger.error(f"❌ STAGE 1: No description column found in {sheet_name}")
                all_results[sheet_name] = {"error": f"No description column found in {sheet_name}"}
                continue
            
            unit_col = stage1.find_unit_column(df)
            logger.info(f"✅ STAGE 1: Found description column: {desc_col}, unit column: {unit_col or 'None'}")
            
            chunks, header_registry = stage1.create_sequential_chunks_enhanced(df, desc_col, unit_col)
            
            if not chunks:
                logger.warning(f"❌ STAGE 1: No chunks created for {sheet_name}")
                all_results[sheet_name] = {"error": f"No processable chunks found in {sheet_name}"}
                continue
            
            if not header_registry:
                logger.warning(f"❌ STAGE 1: No headers registered for {sheet_name}")
                all_results[sheet_name] = {"error": f"No headers found in {sheet_name}"}
                continue
            
            logger.info(f"✅ STAGE 1 COMPLETE: Created {len(chunks)} chunks and registered {len(header_registry)} headers for {sheet_name}")
            
            # STAGE 2: AI-powered header verification
            logger.info(f"🤖 STAGE 2: Starting AI header verification for {sheet_name} - verifying {len(header_registry)} headers")
            
            verified_headers = await stage2.verify_headers(header_registry)
            
            if not verified_headers:
                logger.warning(f"❌ STAGE 2: No verified headers for {sheet_name}")
                all_results[sheet_name] = {"error": f"No verified headers found in {sheet_name}"}
                continue
            
            logger.info(f"✅ STAGE 2 COMPLETE: Verified {len(verified_headers)} headers for {sheet_name}")
            
            # FIXED: Debug chunk ID mapping between Stage 2 and Stage 3
            logger.info(f"🔍 STAGE 2->3 MAPPING DEBUG:")
            verified_header_map = {vh.chunk_id: vh for vh in verified_headers}
            chunk_ids_from_stage1 = [chunk.chunk_id for chunk in chunks]
            chunk_ids_from_stage2 = list(verified_header_map.keys())
            
            logger.info(f"  - Chunk IDs from Stage 1: {chunk_ids_from_stage1}")
            logger.info(f"  - Chunk IDs from Stage 2: {chunk_ids_from_stage2}")
            
            missing_mappings = [cid for cid in chunk_ids_from_stage1 if cid not in verified_header_map]
            if missing_mappings:
                logger.warning(f"⚠️ Missing header context for chunks: {missing_mappings}")
            
            # STAGE 3: AI context inheritance for line items
            logger.info(f"🧠 STAGE 3: Starting AI context inheritance for {sheet_name} - processing {len(chunks)} chunks")
            tasks = []
            processed_chunks = 0
            
            for chunk in chunks:
                header_context = verified_header_map.get(chunk.chunk_id)
                if header_context:
                    logger.info(f"  - Creating task for chunk {chunk.chunk_id} with {len(chunk.items)} items")
                    tasks.append(stage3.extract_chunk_details_enhanced(
                        chunk_info={'chunk_id': chunk.chunk_id}, 
                        verified_header=header_context, 
                        chunk_items=chunk.items
                    ))
                    processed_chunks += 1
                else:
                    logger.warning(f"⚠️ No header context found for chunk {chunk.chunk_id} - skipping")
            
            if not tasks:
                logger.warning(f"❌ STAGE 3: No tasks created for {sheet_name}")
                all_results[sheet_name] = {"error": f"No processable tasks found in {sheet_name}"}
                continue
            
            logger.info(f"🚀 STAGE 3: Executing {len(tasks)} parallel tasks for {processed_chunks} chunks")
            
            # Execute all Stage 3 tasks in parallel
            line_item_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results and handle exceptions
            processed_items = []
            successful_tasks = 0
            failed_tasks = 0
            
            for i, result in enumerate(line_item_results_list):
                if isinstance(result, Exception):
                    logger.error(f"❌ STAGE 3: Task {i+1} failed: {result}")
                    failed_tasks += 1
                    continue
                processed_items.extend(result)
                successful_tasks += 1
            
            logger.info(f"✅ STAGE 3 COMPLETE: {successful_tasks}/{len(tasks)} tasks successful, {failed_tasks} failed")
            
            if not processed_items:
                logger.warning(f"❌ STAGE 3: No items processed for {sheet_name}")
                all_results[sheet_name] = {"error": f"No items successfully processed in {sheet_name}"}
                continue
            
            # Convert the final list of dataclass objects to a list of dictionaries for the frontend
            logger.info(f"📋 FINAL: Converting {len(processed_items)} items to JSON format for {sheet_name}")
            sheet_data_for_json = []
            
            for item in processed_items:
                try:
                    # Create a dictionary from the dataclass
                    item_dict = item.__dict__.copy()
                    
                    # Add the original row data from the DataFrame for the verification grid
                    if item.original_row_index < len(df):
                        original_row_data = df.iloc[item.original_row_index].to_dict()
                        item_dict['original_data'] = {k: v for k, v in original_row_data.items() if pd.notna(v)}
                    else:
                        item_dict['original_data'] = {}
                    
                    sheet_data_for_json.append(item_dict)
                    
                except Exception as e:
                    logger.error(f"❌ Error converting item to JSON: {e}")
                    continue

            all_results[sheet_name] = {
                "data": sheet_data_for_json,
                "stats": {
                    "total_items": len(processed_items),
                    "chunks_processed": processed_chunks,
                    "chunks_total": len(chunks),
                    "headers_verified": len(verified_headers),
                    "headers_registered": len(header_registry),
                    "description_column": desc_col,
                    "unit_column": unit_col,
                    "stage3_success_rate": f"{successful_tasks}/{len(tasks)}"
                }
            }
            
            logger.info(f"🎉 SUCCESS: Sheet {sheet_name} completed - {len(processed_items)} items extracted through 3-stage pipeline")

    except Exception as e:
        logger.error(f"❌ ORCHESTRATOR ERROR: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": str(e)}

    logger.info(f"🎉 ORCHESTRATOR COMPLETE: Processed {len(all_results)} sheets successfully")
    return all_results


def generate_summary_file(corrected_data: List[Dict]) -> BytesIO:
    """
    Takes corrected data and generates the final summary Excel file in memory.
    
    Args:
        corrected_data: List of dictionaries containing corrected BOQ data
        
    Returns:
        BytesIO object containing the Excel file
    """
    logger.info("📊 SUMMARIZER: Generating final summary file...")
    
    try:
        # Convert list of dictionaries to DataFrame
        data_df = pd.DataFrame(corrected_data)
        
        if data_df.empty:
            logger.warning("⚠️ No data provided for summary generation")
            # Create empty summary file
            data_df = pd.DataFrame({"Message": ["No data provided for summary"]})
        
        logger.info(f"📊 Generating summary for {len(data_df)} rows")
        
        # Generate material summary using our summarizer
        summary_df = generate_summary_dataframe(data_df)
        
        # Create an in-memory Excel file to send back to the user
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write the material summary
            summary_df.to_excel(writer, sheet_name='Material_Summary', index=False)
            
            # Also include the original corrected data for reference
            if len(corrected_data) > 0 and "Message" not in data_df.columns:
                data_df.to_excel(writer, sheet_name='Corrected_Data', index=False)
        
        output.seek(0)
        
        logger.info("✅ Successfully generated summary Excel file")
        return output
        
    except Exception as e:
        logger.error(f"❌ Error generating summary file: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Create error file
        error_output = BytesIO()
        error_df = pd.DataFrame({
            "Error": ["Failed to generate summary"],
            "Details": [str(e)],
            "Input_Rows": [len(corrected_data)]
        })
        
        with pd.ExcelWriter(error_output, engine='openpyxl') as writer:
            error_df.to_excel(writer, sheet_name='Error_Report', index=False)
        
        error_output.seek(0)
        return error_output


async def run_complete_boq_processing(file_path: str, selected_sheets: List[str], api_key: str) -> Dict:
    """
    Complete BOQ processing pipeline - from file to extracted data.
    This is the main entry point for the orchestrator.
    
    Args:
        file_path: Path to the Excel file
        selected_sheets: List of worksheet names to process
        api_key: Gemini API key
        
    Returns:
        Dictionary with extraction results and statistics
    """
    logger.info(f"🚀 COMPLETE BOQ PROCESSING: Starting pipeline for {len(selected_sheets)} sheets")
    
    try:
        # Run the extraction pipeline
        extraction_results = await run_extraction_pipeline(file_path, selected_sheets, api_key)
        
        # Calculate overall statistics
        total_items = 0
        total_sheets_processed = 0
        total_errors = 0
        
        for sheet_name, result in extraction_results.items():
            if "error" in result:
                total_errors += 1
                logger.warning(f"❌ Sheet {sheet_name} failed: {result['error']}")
            else:
                total_sheets_processed += 1
                if "data" in result:
                    total_items += len(result["data"])
                elif isinstance(result, list):
                    total_items += len(result)
                logger.info(f"✅ Sheet {sheet_name} processed: {len(result.get('data', []))} items")
        
        success_rate = total_sheets_processed / len(selected_sheets) if selected_sheets else 0
        
        logger.info(f"🎉 PIPELINE COMPLETE: {total_sheets_processed}/{len(selected_sheets)} sheets successful ({success_rate:.1%})")
        logger.info(f"📊 TOTAL ITEMS EXTRACTED: {total_items}")
        
        return {
            "extraction_results": extraction_results,
            "overall_stats": {
                "total_sheets_requested": len(selected_sheets),
                "total_sheets_processed": total_sheets_processed,
                "total_items_extracted": total_items,
                "total_errors": total_errors,
                "processing_success_rate": success_rate
            }
        }
        
    except Exception as e:
        logger.error(f"❌ COMPLETE BOQ PROCESSING ERROR: {e}")
        return {
            "error": str(e),
            "overall_stats": {
                "total_sheets_requested": len(selected_sheets),
                "total_sheets_processed": 0,
                "total_items_extracted": 0,
                "total_errors": len(selected_sheets),
                "processing_success_rate": 0
            }
        }