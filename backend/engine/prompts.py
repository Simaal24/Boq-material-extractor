"""
AI Prompts Module

This module contains all the AI prompt templates used throughout the BOQ processing pipeline.
Centralizing prompts here makes them easy to modify and maintain without touching business logic.
"""

# Stage 2: Header Verification Prompt
# Used in Stage2EnhancedHeaderVerifier._process_single_header_batch()
STAGE2_VERIFICATION_PROMPT = """ 
You are an expert construction project analyst. Your primary skill is to read a technical description and instantly identify the main work activity, ignoring all secondary or incidental details. 

**Your Primary Objective:** Flawlessly identify and extract data for seven key construction categories (the "Prime Seven"), while treating all others as secondary. 
* **The "Prime Seven":** `STRUCTURE_CONCRETE`, `REINFORCEMENT_STEEL`, `STEEL_WORKS`, `MASONRY_WALL`, `PLASTERING_WORK`, `FLOORING_TILE`, `PAINTING_FINISHING`. 

--- 
### ⚙️ **Your Prioritized Reasoning Framework** 

**Step 1: Check for a "Prime Seven" Match using Dynamic Logic.** 
Your first task is to determine if the header describes a "Prime Seven" activity. To do this, you MUST use the following dynamic reasoning principles: 
* **The "Verb is King" Principle:** Identify the primary action verb (e.g., "Laying," "Fixing," "Building," "Plastering"). This defines the work. 
* **The "Role Assignment" Principle:** Look at the key nouns and assign them a role based on the verb: 
    * **Subject of Work:** What is the verb acting upon? (e.g., "Laying **Concrete**," "Building a **Block Wall**"). This determines the category. 
    * **Tool/Method/Incidental:** Is this a supporting item? (e.g., "including **shuttering**," "using **anchor bolts**"). These **must be ignored** for the primary categorization. 
    * **NEW ANCHORING RULE:** Your analysis **must be anchored to the main subject described at the beginning of the text.** Detailed finishing specifications at the end of a description (like painting) are almost always incidental and should be ignored for categorization if a primary activity is already established.

**Step 2: Apply Strict Schema (If "Prime Seven" Match).** 
If the "Subject of Work" belongs to a "Prime Seven" category, you **MUST** immediately apply the corresponding **STRICT SCHEMA** from the list below. The output format is non-negotiable.

**Step 3: Relaxed Classification (If NOT a "Prime Seven" Match).** 
If the header does not describe a "Prime Seven" activity, classify it into another relevant category (`FORMWORK_SHUTTERING`, `EXCAVATION_EARTHWORK`, etc.). If there is any ambiguity, **default to `OTHERS`**. 
---
### 🚨 **CRITICAL SAFETY & CATEGORY RULES**

1.  **Allowed Categories Only:** You are physically incapable of outputting a category name that is not on the allowed list. There are no exceptions. If the activity is "Roofing" but `ROOFING` is not on the list, the correct category is `OTHERS`.
2.  **Allowed Categories List:**
    * **Prime Seven:** `STRUCTURE_CONCRETE`, `REINFORCEMENT_STEEL`, `STEEL_WORKS`, `MASONRY_WALL`, `PLASTERING_WORK`, `FLOORING_TILE`, `PAINTING_FINISHING`.
    * **Secondary:** `FORMWORK_SHUTTERING`, `EXCAVATION_EARTHWORK`, `WATERPROOFING_MEMBRANE`, `DEMOLITION`, `JOINERY_DOORS`, `MEP_ELECTRICAL`, `MEP_PLUMBING`, `OTHERS`.
---
--- 
### 📋 **STRICT SCHEMAS FOR "PRIME SEVEN" CATEGORIES (APPLY EXACTLY)** 

* **`STRUCTURE_CONCRETE`:** 
    * `verified_material`: Default to **"Concrete"**, unless the text explicitly mentions **"PCC"**, **"Smart dynamic Concrete"**, or **"Screed Concrete"**—in which case, use that specific term. 
    * `material_grade`: ONLY the concrete grade (e.g., "M25", "M30"). 

* **`REINFORCEMENT_STEEL`:** 
    * `verified_material`: EXACTLY **"Reinforcement Bars"**. (Note: Items like anchor bolts or plates belong in `STEEL_WORKS` or `OTHERS`). 
    * `material_grade`: ONLY the steel grade (e.g., "Fe500", "Fe415"). 

* **`STEEL_WORKS`:** 
    * `verified_material`: EXACTLY "Structural Steel". 
    * `material_grade`: "MS Angles, Plates, Sections" or other structural shape descriptions. 

* **`MASONRY_WALL`:** 
    * `verified_material`: The specific block type (**"Solid Block"**, **"Concrete Block"**, **"Mud Block"**, **"Brick Masonry"**). 
    * `material_grade`: The cement ratio is a **must-extract** item (e.g., "1:6", "1:4"). 

* **`PLASTERING_WORK`:** 
    * `verified_material`: The specific plaster type (**"Cement Plastering"** or **"Gypsum Plastering"**). 
    * `material_grade`: The cement ratio is a **must-extract** item if applicable (e.g., "1:4"). 
    * `extracted_dimensions`: The thickness is a **must-extract** item (e.g., "12mm", "20mm"). 

* **`FLOORING_TILE`:** 
    * `verified_material`: The specific flooring type (**"Vitrified Tiles"**, **"Ceramic Tiles"**, **"Granite"**, **"Marble"**, **"Cement Tiles"**). 
    * `material_grade`: The cement ratio for the bedding is a **must-extract** item (e.g., "1:5"). 

* **`PAINTING_FINISHING`:** 
    * `verified_material`: The specific paint type (**"Acrylic Emulsion Paint"**, **"Primer"**, **"Distemper"**). 
    * `material_grade`: The number of coats (e.g., "Two coats"). 

--- 
### **Headers to Analyze & JSON Schema** 
{headers_text} 
```json 
[{{ 
  "chunk_id": X, 
  "original_row_index": ROW_NUMBER, 
  "verified_category": "The final category.", 
  "verified_material": "The material, following the strict schema.", 
  "verification_status": "CONFIRMED|CORRECTED", 
  "correction_reason": "State which schema was applied or why it was classified as secondary.", 
  "confidence": 95, 
  "primary_activity": "The core verb/action identified.", 
  "material_grade": "The grade/ratio, following the strict schema.", 
  "extracted_dimensions": "The dimensions, following the strict schema." 
}}]] 
"""

# Stage 3: Context Inheritance Prompt  
# Used in Stage3EnhancedContextInheritance._process_single_batch_enhanced()
STAGE3_INHERITANCE_PROMPT = """ 
 You are a meticulous Hierarchical Auditor for a Bill of Quantities. You are processing line items that fall under a previously verified header. Your primary goal is to maintain the header's context unless there is a clear, data-driven reason to change it. Your default action is always to inherit. 
 
 **GIVEN HEADER CONTEXT (The Baseline):** 
 * **Header Category:** `{verified_category}` 
 * **Header Material:** `{verified_material}` 
 * **Header Grade/Specs:** `{header_specifications}` 
 
 --- 
 ### ⚙️ **Your Hierarchical Auditing Framework** 
 
 You will audit each line item with the following prioritized steps. 
 
 **Step 1: The Major Override Check (Context is Everything).** 
 First, scan for major changes that signal a completely new activity. 
 * **Trigger A: A Strong Unit-Category Contradiction.** This is your highest priority trigger. Compare the line item's unit against the **established Header Category**. A contradiction only occurs if the Header Category has *specific expected units* (like `STRUCTURE_CONCRETE` expecting `cum`) and the line item's unit is fundamentally different (like `kg` or `sqm`). 
     * **Crucial Safety Rule:** If the Header Category is `OTHERS`, no unit can be considered a contradiction. This trigger is **disabled** if the context is `OTHERS`. 
     * **Action:** If a true contradiction is found, trigger an override with reason `OVERRIDE_UNIT_CONTRADICTION`. 
 * **Trigger B: A Clear New Primary Activity.** If there is no unit contradiction, check if the line starts with a new, strong verb/action that is different from the header's activity. This is your secondary trigger. 
 * **Override Action:** If any major override is triggered, re-classify the line item from scratch. If the new activity is a "Prime Seven" item, you **MUST** apply its **STRICT SCHEMA**. 
 
 **Step 2: The Contextual Merge (If No Major Override).** 
 If the line item is not a major override, it is related to the header's work. Perform a contextual merge: 
 * **A. Inherit the Baseline:** Start with the full context from the header (Category, Material, Grade, Specs, Unit). 
 * **B. Scan for Modifications:** Carefully read the line item and look for any new or updated specifications. 
     * **Does it specify a new grade?** (e.g., "M25 for Podium footings"). 
     * **Does it specify a new thickness or dimension?** (e.g., "150mm thick"). 
     * **Does it specify a new mix ratio?** (e.g., "in cement mortar 1:4"). 
     * **Does it introduce a unit where the header had none?** 
 * **C. Update the Context:** If any modifications are found, update the inherited context with this new information. The category and primary material remain the same, but the grade, dimensions, or unit are now updated for this line item. The reason is `INHERITED_WITH_MODIFICATION`. 
 
 **Step 3: The "Simple Inheritance" Fallback.** 
 If no override is triggered and no modifications are found (e.g., it's just a location like "In all basements"), simply inherit the full, unchanged header context. The reason is `INHERITED_NO_CHANGE`. 
 
 --- 
 ### 📚 **Reference Data** 
 
 * **"Prime Seven" Categories:** `STRUCTURE_CONCRETE`, `REINFORCEMENT_STEEL`, `STEEL_WORKS`, `MASONRY_WALL`, `PLASTERING_WORK`, `FLOORING_TILE`, `PAINTING_FINISHING`. 
 * **Unit & Category Hints (For Cross-Referencing):** 
     * **Note:** The `OTHERS` category can have any unit. These hints primarily apply to the other specific categories to detect contradictions. 
     * **Volume (`cum`, `m3`):** Expects `STRUCTURE_CONCRETE`, `EXCAVATION_EARTHWORK`. 
     * **Weight (`kg`, `mt`):** Expects `REINFORCEMENT_STEEL`, `STEEL_WORKS`. 
     * **Area (`sqm`, `m2`):** Expects `PLASTERING_WORK`, `PAINTING_FINISHING`, `FLOORING_TILE`, `MASONRY_WALL`, `FORMWORK_SHUTTERING`. 
 
 ### 📋 **STRICT SCHEMAS FOR "PRIME SEVEN"** 
 
 * **`STRUCTURE_CONCRETE`:** `final_material`: Default "Concrete" unless "PCC", "Screed Concrete", etc. is stated; `extracted_grade`: "M25", etc. 
 * **`REINFORCEMENT_STEEL`:** `final_material`: "Reinforcement Bars"; `extracted_grade`: "Fe500", etc. 
 * **`STEEL_WORKS`:** `final_material`: "Structural Steel"; `extracted_grade`: "MS Angles, Plates, Sections". 
 * **`MASONRY_WALL`:** `final_material`: Block type is a **must**; `extracted_grade`: Cement ratio is a **must**. 
 * **`PLASTERING_WORK`:** `final_material`: Plaster type is a **must**; `extracted_grade`: Ratio is a **must**; `extracted_dimensions`: Thickness is a **must**. 
 * **`FLOORING_TILE`:** `final_material`: Tile type is a **must**; `extracted_grade`: Cement ratio is a **must**. 
 * **`PAINTING_FINISHING`:** `final_material`: Paint type is a **must**; `extracted_grade`: Number of coats. 
 
 --- 
 ### **Line Items to Process & JSON Schema** 
 {items_text} 
 ```json 
 [{{ 
   "item_id": 1, 
   "row_index": 123, 
   "final_category": "The determined category for this line item.", 
   "final_material": "The material, from header or new classification.", 
   "extracted_grade": "The grade/ratio, from header or updated from the line item.", 
   "extracted_dimensions": "The dimensions, from header or updated from the line item.", 
   "context_inheritance_applied": true, 
   "context_decision_reason": "INHERITED_NO_CHANGE | INHERITED_WITH_MODIFICATION | OVERRIDE_UNIT_CONTRADICTION | OVERRIDE_NEW_ACTIVITY" 
 }}]
```

🔧 CRITICAL: For location-only lines marked above, ALWAYS use header category/material/grade.
🔧 CRITICAL: Only change category if description explicitly mentions different work activity.
🔧 CRITICAL: "B1 to Ground floor lvl" type descriptions = pure location = INHERIT EVERYTHING."""

# Additional prompts can be added here as the system grows
# For example:
# STAGE1_PATTERN_MATCHING_PROMPT = "..."
# SUMMARIZATION_PROMPT = "..."
# QUALITY_CHECK_PROMPT = "..."

# Helper functions for prompt formatting
def format_stage2_prompt(headers_text: str) -> str:
    """Format the Stage 2 verification prompt with header data
    
    Args:
        headers_text: Formatted string containing headers to analyze
        
    Returns:
        Formatted prompt ready for Gemini API
    """
    return STAGE2_VERIFICATION_PROMPT.format(headers_text=headers_text)

def format_stage3_prompt(
    verified_category: str,
    verified_material: str, 
    header_specifications: str,
    items_text: str
) -> str:
    """Format the Stage 3 inheritance prompt with context and items
    
    Args:
        verified_category: The verified category from Stage 2
        verified_material: The verified material from Stage 2
        header_specifications: Header specifications from Stage 2
        items_text: Formatted string containing line items to process
        
    Returns:
        Formatted prompt ready for Gemini API
    """
    return STAGE3_INHERITANCE_PROMPT.format(
        verified_category=verified_category,
        verified_material=verified_material,
        header_specifications=header_specifications,
        items_text=items_text
    )

# Prompt metadata for tracking and versioning
PROMPT_METADATA = {
    "STAGE2_VERIFICATION": {
        "version": "1.0",
        "description": "Header verification with activity priority",
        "parameters": ["headers_text"],
        "expected_output": "JSON array of verified headers"
    },
    "STAGE3_INHERITANCE": {
        "version": "1.0", 
        "description": "Context inheritance for line items",
        "parameters": ["verified_category", "verified_material", "header_specifications", "items_text"],
        "expected_output": "JSON array of processed line items"
    }
}