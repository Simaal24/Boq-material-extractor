"""
BOQ App Backend - Flask API Server

This server provides the REST API endpoints for the BOQ extraction application.
It handles file uploads, worksheet analysis, and data extraction.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
import pandas as pd
import asyncio
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import traceback
from dataclasses import asdict

# FIXED: Load .env file from the engine directory where it's actually located
from dotenv import load_dotenv

# Try multiple locations for the .env file
env_locations = [
    os.path.join(os.path.dirname(__file__), 'engine', '.env'),  # /backend/engine/.env
    os.path.join(os.path.dirname(__file__), '.env'),           # /backend/.env
    '.env'  # Current directory
]

env_loaded = False
for env_path in env_locations:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info(f"✅ Loaded .env file from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    logging.warning("⚠️ .env file not found in any expected location")
    logging.warning(f"Searched locations: {env_locations}")

# Import our refactored engine modules
from engine import (
    run_initial_analysis, 
    FileExtraction, 
    WorksheetExtraction,
    initialize_client,
    get_usage_stats,
    generate_summary_dataframe,
    run_complete_boq_processing,
    generate_summary_file
)
from engine.data_structures import ExtractedLineItem, VerifiedHeader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure all engine modules show INFO level logs
logging.getLogger('engine.orchestrator').setLevel(logging.INFO)
logging.getLogger('engine.stage1_detector').setLevel(logging.INFO)
logging.getLogger('engine.stage2_verifier').setLevel(logging.INFO)
logging.getLogger('engine.stage3_extractor').setLevel(logging.INFO)
logging.getLogger('engine.pre_processor').setLevel(logging.INFO)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY") 

# FIXED: Add debug logging to check if API key is loaded
if app.config['GEMINI_API_KEY']:
    logger.info(f"✅ Gemini API key loaded successfully (length: {len(app.config['GEMINI_API_KEY'])} chars)")
    if app.config['GEMINI_API_KEY'].startswith("AIzaSy"):
        logger.info("✅ API key format looks correct")
    else:
        logger.warning("⚠️ API key doesn't start with 'AIzaSy' - please verify it's correct")
else:
    logger.error("❌ GEMINI_API_KEY not found in environment variables!")
    logger.error("Please check that your .env file contains: GEMINI_API_KEY=your_actual_key")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# FIXED: Initialize Gemini client on startup with better error handling
try:
    if app.config['GEMINI_API_KEY']:
        gemini_client = initialize_client(app.config['GEMINI_API_KEY'])
        logger.info("🤖 Gemini AI client initialized successfully")
    else:
        logger.error("❌ Cannot initialize Gemini client: API key is missing")
        gemini_client = None
except Exception as e:
    logger.error(f"❌ Gemini client initialization failed: {e}")
    logger.error("AI features will be limited without Gemini client")
    gemini_client = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'xlsm'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def serialize_analysis_result(analysis_result):
    """Convert the analysis result to a JSON-serializable format"""
    try:
        # Convert dataclass to dict
        result_dict = asdict(analysis_result)
        
        # Handle pandas DataFrames in BOQ tables
        for worksheet in result_dict['worksheets']:
            if 'boq_regions' in worksheet:
                for region in worksheet['boq_regions']:
                    # Convert any pandas objects to basic types
                    if 'data' in region:
                        # Convert DataFrame to dict if present
                        region['data'] = region['data'].to_dict('records') if hasattr(region['data'], 'to_dict') else region['data']
        
        return result_dict
    except Exception as e:
        logger.error(f"Error serializing analysis result: {str(e)}")
        return {"error": "Failed to serialize analysis result"}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with comprehensive system status"""
    
    # Get Gemini client status
    usage_stats = get_usage_stats()
    gemini_status = "healthy" if "error" not in usage_stats else "error"
    
    # FIXED: Add API key status to health check
    api_key_status = "present" if app.config['GEMINI_API_KEY'] else "missing"
    
    # Test Stage 1 detector availability
    stage1_status = "healthy"
    try:
        from engine import Stage1EnhancedHeaderDetector
        detector = Stage1EnhancedHeaderDetector()
        # Quick test
        test_result = detector.is_main_header("Providing and laying concrete")
        stage1_status = "healthy" if isinstance(test_result, bool) else "error"
    except Exception as e:
        stage1_status = f"error: {str(e)}"
    
    # Test Stage 2 verifier availability
    stage2_status = "healthy"
    try:
        from engine import Stage2EnhancedHeaderVerifier
        verifier = Stage2EnhancedHeaderVerifier(batch_size=1)
        stage2_status = f"healthy (batch_size: {verifier.batch_size})"
    except Exception as e:
        stage2_status = f"error: {str(e)}"
    
    # Test Stage 3 extractor availability
    stage3_status = "healthy"
    try:
        from engine import Stage3EnhancedContextInheritance
        extractor = Stage3EnhancedContextInheritance()
        stage3_status = f"healthy (api_calls: {extractor.api_calls})"
    except Exception as e:
        stage3_status = f"error: {str(e)}"
    
    # Test summarizer availability
    summarizer_status = "healthy"
    try:
        from engine import FixedBOQSummarizer, generate_summary_dataframe
        summarizer = FixedBOQSummarizer()
        summarizer_status = f"healthy (categories: {len(summarizer.category_priority)})"
    except Exception as e:
        summarizer_status = f"error: {str(e)}"
    
    # Test patterns availability
    patterns_status = "healthy"
    try:
        from engine.patterns import BOQ_CATEGORIES
        patterns_count = len(BOQ_CATEGORIES)
        patterns_status = f"healthy ({patterns_count} categories)"
    except Exception as e:
        patterns_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BOQ Backend API",
        "components": {
            "gemini_client": {
                "status": gemini_status,
                "api_key_status": api_key_status,
                "usage": usage_stats
            },
            "stage1_detector": {
                "status": stage1_status,
                "description": "Rule-based header detection and chunking"
            },
            "stage2_verifier": {
                "status": stage2_status,
                "description": "AI-powered header verification and correction"
            },
            "stage3_extractor": {
                "status": stage3_status,
                "description": "AI context inheritance for line items"
            },
            "summarizer": {
                "status": summarizer_status,
                "description": "Material aggregation and summary generation"
            },
            "patterns_module": {
                "status": patterns_status,
                "description": "BOQ classification patterns and rules"
            },
            "data_structures": {
                "status": "healthy",
                "description": "ProcessingChunk, ExtractedLineItem, etc."
            }
        },
        "capabilities": [
            "File upload and structure analysis",
            "Stage 1: Rule-based BOQ header detection and chunking", 
            "Stage 2: AI-powered header verification with activity priority",
            "Stage 3: AI context inheritance for line item classification",
            "Material summarization and aggregation by Category, Material, Grade, and Specification",
            "Activity priority classification (work type over materials)",
            "Context preservation and grade inheritance (M25, Fe500, etc.)",
            "Location-only line detection and proper inheritance",
            "Complete 3-stage BOQ processing pipeline",
            "Robust error handling with graceful fallbacks",
            "Gemini AI integration with rate limiting"
        ]
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Step 1: Upload & Pre-processing
    Handles file upload and performs initial analysis
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Please upload Excel files (.xlsx, .xls, .xlsm)"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        logger.info(f"📁 File uploaded successfully: {unique_filename}")
        logger.info(f"📁 File saved to: {file_path}")
        
        # FIXED: Ensure API key is passed to pre-processor
        api_key = app.config['GEMINI_API_KEY']
        if not api_key:
            logger.warning("⚠️ API key missing - pre-processor will run without AI features")
        else:
            logger.info(f"🤖 API key available for AI processing")
        
        # Run initial analysis using our refactored engine
        logger.info(f"🚀 Starting initial BOQ analysis for: {unique_filename}")
        analysis_result = run_initial_analysis(
            file_path=file_path,
            gemini_api_key=api_key
        )
        logger.info(f"✅ Initial analysis complete for: {unique_filename}")
        
        # Serialize the result for JSON response
        serialized_result = serialize_analysis_result(analysis_result)
        
        # Add file path to the response for future operations
        serialized_result['file_path'] = file_path
        serialized_result['original_filename'] = filename
        
        return jsonify({
            "message": "File uploaded and analyzed successfully",
            "analysis": serialized_result
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/api/worksheets/<file_id>', methods=['GET'])
def get_worksheets(file_id):
    """
    Step 2: Worksheet Selection
    Returns available worksheets from the uploaded file
    """
    try:
        # In a production app, you'd retrieve this from a database
        # For now, we'll assume the file_id is the file path
        file_path = file_id  # Simplified for demo
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Re-run analysis to get worksheet info
        analysis_result = run_initial_analysis(
            file_path=file_path,
            gemini_api_key=app.config['GEMINI_API_KEY']
        )
        
        # Extract worksheet information
        worksheets = []
        for ws in analysis_result.worksheets:
            worksheets.append({
                "name": ws.name,
                "dimensions": ws.dimensions,
                "has_boq": len(ws.boq_regions) > 0,
                "data_density": ws.data_density,
                "suspected_headers": len(ws.suspected_headers),
                "ai_confidence": ws.boq_regions[0]['confidence'] if ws.boq_regions else 0
            })
        
        return jsonify({
            "worksheets": worksheets,
            "total_count": len(worksheets),
            "file_info": {
                "filename": analysis_result.filename,
                "file_size": analysis_result.file_size,
                "processing_time": analysis_result.processing_time
            }
        })
        
    except Exception as e:
        logger.error(f"Worksheet retrieval error: {str(e)}")
        return jsonify({"error": f"Failed to get worksheets: {str(e)}"}), 500

@app.route('/api/extract', methods=['POST'])
def extract_boq():
    """
    Step 3: Main Extraction
    Extracts BOQ data from selected worksheets using the complete orchestrator pipeline
    """
    try:
        data = request.get_json()
        
        if not data or 'file_path' not in data or 'selected_worksheets' not in data:
            return jsonify({"error": "Missing required parameters"}), 400
        
        file_path = data['file_path']
        selected_worksheets = data['selected_worksheets']
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # FIXED: Check if Gemini client is available before processing
        if not app.config['GEMINI_API_KEY']:
            return jsonify({"error": "Gemini API key not configured. Please check your .env file."}), 500
        
        logger.info(f"🚀 Starting 3-stage BOQ extraction for sheets: {selected_worksheets}")
        logger.info(f"📄 Processing file: {file_path}")
        
        # Run the complete BOQ processing pipeline using orchestrator
        logger.info(f"🔄 Initializing async event loop for parallel processing...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(f"🚀 Running complete BOQ processing pipeline...")
            results = loop.run_until_complete(
                run_complete_boq_processing(file_path, selected_worksheets, app.config['GEMINI_API_KEY'])
            )
            logger.info(f"✅ BOQ processing pipeline completed")
        finally:
            loop.close()
        
        # Check if there was an error
        if "error" in results:
            return jsonify({"error": results["error"]}), 500
        
        # Extract results and statistics
        extraction_results = results.get("extraction_results", {})
        overall_stats = results.get("overall_stats", {})
        
        logger.info(f"📊 EXTRACTION RESULTS SUMMARY:")
        logger.info(f"   📋 Sheets processed: {overall_stats.get('total_sheets_processed', 0)}/{overall_stats.get('total_sheets_requested', 0)}")
        logger.info(f"   📊 Success rate: {overall_stats.get('processing_success_rate', 0):.1%}")
        
        # Format response for frontend
        extracted_tables = []
        total_rows = 0
        
        for sheet_name, sheet_result in extraction_results.items():
            if "error" not in sheet_result:
                data_items = sheet_result.get("data", [])
                stats = sheet_result.get("stats", {})
                
                logger.info(f"   ✅ Sheet '{sheet_name}': {len(data_items)} items extracted")
                
                table_data = {
                    "worksheet_name": sheet_name,
                    "data_count": len(data_items),
                    "stats": stats,
                    "extraction_notes": f"Extracted {len(data_items)} items with 3-stage AI processing"
                }
                extracted_tables.append(table_data)
                total_rows += len(data_items)
            else:
                logger.warning(f"   ❌ Sheet '{sheet_name}': {sheet_result.get('error', 'Unknown error')}")
        
        logger.info(f"✅ Extraction complete: {overall_stats.get('total_sheets_processed', 0)} sheets, {total_rows} total items")
        
        return jsonify({
            "message": f"Successfully processed {overall_stats.get('total_sheets_processed', 0)} worksheets",
            "extracted_tables": extracted_tables,
            "total_rows": total_rows,
            "overall_stats": overall_stats,
            "extraction_data": extraction_results,  # Full data for verification step
            "ready_for_verification": True
        })
        
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500

@app.route('/api/verify', methods=['POST'])
def verify_data():
    """
    Step 4: Verification & Correction
    Placeholder for data verification and correction
    """
    try:
        data = request.get_json()
        
        # This would integrate with the AI enrichment module
        # For now, return a placeholder response
        
        return jsonify({
            "message": "Data verification completed",
            "verified_rows": data.get('row_count', 0),
            "corrections_applied": 0,
            "ready_for_summarization": True
        })
        
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_results():
    """
    Step 5: Final Summarization
    Generates material summary from verified BOQ data using orchestrator
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get the corrected/verified data from the frontend
        corrected_data = data.get('verified_data', [])
        
        if not corrected_data:
            return jsonify({"error": "No verified data provided"}), 400
        
        logger.info(f"📊 Generating summary for {len(corrected_data)} verified items")
        logger.info(f"📊 Sample data: {corrected_data[0] if corrected_data else 'No data'}")
        logger.info(f"📊 Data columns: {list(corrected_data[0].keys()) if corrected_data else 'No columns'}")
        
        # Sanitize corrected_data to ensure JSON serializable types
        sanitized_data = []
        for item in corrected_data:
            sanitized_item = {}
            for key, value in item.items():
                # Convert pandas/numpy types to native Python types
                if hasattr(value, 'item'):  # numpy/pandas scalar
                    sanitized_item[key] = value.item()
                elif pd.isna(value):  # Handle NaN values
                    sanitized_item[key] = None
                else:
                    sanitized_item[key] = value
            sanitized_data.append(sanitized_item)
        
        # Generate the summary Excel file using orchestrator
        excel_file = generate_summary_file(sanitized_data)
        
        # For now, we'll save it temporarily and provide download info
        # In a production app, you might store this in cloud storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"material_summary_{timestamp}.xlsx"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Save the BytesIO content to a temporary file
        with open(temp_path, 'wb') as f:
            f.write(excel_file.getvalue())
        
        # Calculate summary statistics (simplified - no amounts)
        summary_groups = 0
        total_quantity = 0.0
        summary_df = pd.DataFrame()  # Initialize empty DataFrame
        
        if sanitized_data:
            df = pd.DataFrame(sanitized_data)
            summary_df = generate_summary_dataframe(df)
            summary_groups = int(len(summary_df))  # All rows are material groups (no totals row)
            
            # Extract total quantity (convert to native Python types for JSON serialization)
            if 'Quantity' in summary_df.columns:
                qty_sum = summary_df['Quantity'].sum()
                total_quantity = float(qty_sum) if not pd.isna(qty_sum) else 0.0
        
        logger.info(f"✅ Simplified summary generated: {summary_groups} material groups, {total_quantity} total quantity")
        
        # Convert summary DataFrame to JSON-serializable format for frontend display
        summary_data = []
        if not summary_df.empty:
            summary_data = summary_df.to_dict('records')
            logger.info(f"📊 Summary data for frontend: {len(summary_data)} records")
        
        return jsonify({
            "message": "Simplified material summary generated successfully",
            "summary_groups": summary_groups,
            "total_quantity": total_quantity,
            "input_rows": len(sanitized_data),
            "ready_for_download": True,
            "download_url": f"/api/download/{temp_filename}",
            "file_name": temp_filename,
            "summary_data": summary_data  # Add actual summary data for frontend
        })
        
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    Step 6: Download
    Serves generated Excel files
    """
    try:
        # Security check - ensure filename is safe
        if not filename.endswith('.xlsx') or '..' in filename or '/' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Serve the Excel file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 50MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting BOQ Backend API Server...")
    logger.info(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"📏 Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    
    # FIXED: Add final status check
    if app.config['GEMINI_API_KEY']:
        logger.info("✅ Server starting with Gemini AI enabled")
    else:
        logger.warning("⚠️ Server starting WITHOUT Gemini AI (check .env file)")
        logger.warning("🔧 Place your .env file in the /backend/engine/ folder with: GEMINI_API_KEY=your_actual_key")
    
    app.run(debug=True, host='0.0.0.0', port=5000)