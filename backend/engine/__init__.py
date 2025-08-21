"""
BOQ Engine Package

This package contains all the core processing modules for BOQ extraction and analysis.

Modules:
- patterns: Central patterns and rules for BOQ classification
- data_structures: Data structure blueprints for consistent data flow
- pre_processor: Initial file analysis and header detection
- summarizer: Material summarization and aggregation
"""

# Import key data structures for easy access
from .data_structures import (
    ProcessingChunk,
    RuleResult,
    VerifiedHeader,
    ExtractedLineItem,
    WorksheetExtraction,
    FileExtraction
)

# Import the main processing function
from .pre_processor import run_initial_analysis

# Import Stage 1 detector components
from .stage1_detector import (
    Stage1EnhancedHeaderDetector,
    run_stage1_detection,
    find_description_column,
    find_unit_column
)

# Import Stage 2 verifier components
from .stage2_verifier import (
    Stage2EnhancedHeaderVerifier,
    run_stage2_verification
)

# Import Stage 3 extractor components
from .stage3_extractor import (
    Stage3EnhancedContextInheritance,
    run_stage3_extraction
)

# Import Summarizer components
from .summarizer import (
    FixedBOQSummarizer,
    generate_summary_dataframe
)

# Import Orchestrator components
from .orchestrator import (
    run_extraction_pipeline,
    generate_summary_file,
    run_complete_boq_processing
)

# Make patterns available when they're added
try:
    from .patterns import *
except ImportError:
    # Patterns module not yet created
    pass

# Import Gemini client components
from .gemini_client import initialize_client, get_client, generate_content, get_usage_stats

# Import prompts
from .prompts import (
    STAGE2_VERIFICATION_PROMPT,
    STAGE3_INHERITANCE_PROMPT,
    format_stage2_prompt,
    format_stage3_prompt,
    PROMPT_METADATA
)

__version__ = "1.0.0"
__author__ = "BOQ Processing Team"

# Define what gets imported with "from engine import *"
__all__ = [
    'ProcessingChunk',
    'RuleResult', 
    'VerifiedHeader',
    'ExtractedLineItem',
    'WorksheetExtraction',
    'FileExtraction',
    'run_initial_analysis',
    'Stage1EnhancedHeaderDetector',
    'run_stage1_detection',
    'find_description_column',
    'find_unit_column',
    'Stage2EnhancedHeaderVerifier',
    'run_stage2_verification',
    'Stage3EnhancedContextInheritance',
    'run_stage3_extraction',
    'FixedBOQSummarizer',
    'generate_summary_dataframe',
    'run_extraction_pipeline',
    'generate_summary_file',
    'run_complete_boq_processing',
    'initialize_client',
    'get_client', 
    'generate_content',
    'get_usage_stats',
    'format_stage2_prompt',
    'format_stage3_prompt'
]