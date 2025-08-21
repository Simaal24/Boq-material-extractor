""" BOQ Patterns Module"""
#!/usr/bin/env python3

import re

# ==============================================================================
# ===== MASTER CATEGORY LIST =====
# ==============================================================================
# This dictionary defines all possible output categories for the system.

BOQ_CATEGORIES = {
    "EXCAVATION_EARTHWORK": {"default_material": "Earth/Soil"},
    "STRUCTURE_CONCRETE": {"default_material": "Concrete"},
    "REINFORCEMENT_STEEL": {"default_material": "Steel"},
    "STEEL_WORKS": {"default_material": "Mild Steel"},
    "FORMWORK_SHUTTERING": {"default_material": "Formwork Material"},
    "MASONRY_WALL": {"default_material": "Masonry"},
    "PLASTERING_WORK": {"default_material": "Cement Mortar"},
    "FLOORING_TILE": {"default_material": "Flooring Material"},
    "WATERPROOFING_MEMBRANE": {"default_material": "Waterproofing Material"},
    "PAINTING_FINISHING": {"default_material": "Paint"},
    "JOINERY_DOORS": {"default_material": "Wood/Steel"},
    "GLAZING": {"default_material": "Glazing System"},
    "MEP_ELECTRICAL": {"default_material": "Electrical Components"},
    "MEP_PLUMBING": {"default_material": "Plumbing Components"},
    "DEMOLITION": {"default_material": "Demolition Debris"},
    "OTHERS": {"default_material": "Mixed"}
}


# ==============================================================================
# ===== CORE ACTIVITY & MATERIAL PATTERNS (Used in Stage 1) =====
# ==============================================================================

# -- Primary action verbs to identify potential headers --
PRIMARY_VERBS = {
    "DEMOLITION": {"variants": ["demolition of", "demolition", "dismantling", "breaking"],"secondary_actions": {}},
    "CONSTRUCTION": {"variants": ["construction of"], "secondary_actions": {"using": ["OTHERS", "MASONRY_WALL"]}},
    "PLASTERING": {"variants": ["plastering"], "secondary_actions": {"with": ["PLASTERING_WORK"]}},
    "PROVIDING": {
        "variants": ["providing", "providing and", "providing &", "providing,"],
        "secondary_actions": {
            "fabricating": ["REINFORCEMENT_STEEL", "STEEL_WORKS"], "laying": ["STRUCTURE_CONCRETE", "FLOORING_TILE", "WATERPROOFING_MEMBRANE"],
            "fixing": ["REINFORCEMENT_STEEL", "STEEL_WORKS", "JOINERY_DOORS", "FALSE_CEILING", "GLAZING"],
            "applying": ["PAINTING_FINISHING", "WATERPROOFING_MEMBRANE"],"constructing": ["MASONRY_WALL"],
            "painting": ["PAINTING_FINISHING"], "anchoring": ["OTHERS"],"making": ["STRUCTURE_CONCRETE"],"grouting": ["OTHERS"],
        }
    },
    "SUPPLYING": {
        "variants": ["supplying", "supply and", "supply,", "supply &", "supply and installation"],
        "secondary_actions": {
            "receiving": ["REINFORCEMENT_STEEL", "MEP_ELECTRICAL"],
            "fixing": ["OTHERS", "REINFORCEMENT_STEEL"],"fabricating": ["REINFORCEMENT_STEEL", "STEEL_WORKS", "MEP_ELECTRICAL"],
            "installation": ["MEP_ELECTRICAL", "MEP_PLUMBING"],"erecting": ["STEEL_WORKS", "FORMWORK_SHUTTERING", "MEP_ELECTRICAL"],
            "injecting": ["WATERPROOFING_MEMBRANE"],"installing": ["MEP_PLUMBING", "MEP_ELECTRICAL"],
            "testing": ["MEP_PLUMBING", "MEP_ELECTRICAL"],"commissioning": ["MEP_PLUMBING", "MEP_ELECTRICAL"],
            "placing": ["STRUCTURE_CONCRETE", "REINFORCEMENT_STEEL"],"fitting": ["REINFORCEMENT_STEEL"],
            "laying": ["MEP_ELECTRICAL"],
        }
    },
    "WIRING": {"variants": ["wiring for", "wiring of"], "secondary_actions": {}},
    "EXCAVATION": {"variants": ["excavation", "excavation in", "earth work excavation"], "secondary_actions": {"in": ["EXCAVATION_EARTHWORK"]}},
    "BACKFILLING": {"variants": ["back filling", "backfilling"], "secondary_actions": {}},
    "CENTERING": {"variants": ["centering and shuttering", "centering & shuttering"], "secondary_actions": {}},
    "PUMPING": {"variants": ["pumping out water"], "secondary_actions": {}},
}

# -- Keywords to identify the primary material in a description --
MATERIAL_INDICATORS = {
    "MEP_ELECTRICAL": {"primary": ["cable", "wiring", "conduit", "mcb", "db", "light fixtures", "cable tray", "earth station"], "secondary": ["xlpe", "pvc", "frls", "sqmm", "sockets", "switches", "gi", "power cables", "terminations"]},
    "DEMOLITION": {"primary": ["demolition", "dismantling", "breaking"], "secondary": ["existing", "debris"]},
    "CONCRETE": {"primary": ["concrete", "pcc", "rcc", "precast", "grade slab"], "secondary": ["grade", "slab", "m20", "m25", "m10"]},
    "STEEL": {"primary": ["steel", "reinforcement", "tmt"], "secondary": ["bars", "fe", "splicing", "coupler", "fe 550"]},
    "STEEL_STRUCTURAL": {"primary": ["ms door", "structural steel", "fire rated steel door"], "secondary": ["angles", "crca sheet"]},
    "MASONRY": {"primary": ["masonry", "brick", "block"], "secondary": ["solid", "aac", "wall"]},
    "FORMWORK": {"primary": ["form work", "formwork", "shuttering", "centering and shuttering"], "secondary": ["mivan", "aluminium", "staging", "system formworks"]},
    "PLASTERING": {"primary": ["plaster", "cement mortar"], "secondary": ["sand face", "gypsum", "internal"]},
    "FLOORING": {"primary": ["granite", "tiles", "flooring", "vitrified", "double charged vitrified"], "secondary": ["polished", "skirting", "800 x 800"]},
    "PAINTING": {"primary": ["paint", "emulsion", "primer", "putty"], "secondary": ["exterior", "acrylic", "two coats"]},
    "WATERPROOFING": {"primary": ["waterproofing", "hdpe membrane", "polyurethane coating", "geotextile"], "secondary": ["elastomeric", "fosroc"]},
    "JOINERY": {"primary": ["teak", "hard wood", "flush shutter", "door frame", "steel door"], "secondary": ["engineered", "fire rated"]},
    "GLAZING": {"primary": ["glazed doors", "glazed windows", "aluminium glazed"], "secondary": ["sliding shutters", "toughened glass", "pvc mosquito mesh"]},
    "ANTI-TERMITE": {"primary": ["anti termite", "anti-termite", "termite treatment"], "secondary": ["chlorpyriphos", "is 6315"]},
    "POLYSHEET": {"primary": ["polythene paper", "polythene sheet"], "secondary": ["micron", "gauge"]},
    "DEWATERING": {"primary": ["pumping out water", "dewatering"], "secondary": ["slurry", "pumps"]},
    "EXCAVATION": {"primary": ["excavation", "earth work"], "secondary": ["hard rock", "footings", "ordinary soil", "back filling"]},
    "PIPES": {"primary": ["hume pipe", "hdpe pipe", "dwc pipe", "gi pipe"], "secondary": ["np 2", "sn8", "class c"]},
    "GROUT": {"primary": ["grout", "grouting"], "secondary": ["epoxy based"]},
    "COATING": {"primary": ["coating", "pu floor coating"], "secondary": ["florthane", "polyurethane resin"]},
    "FALSE_CEILING": {"primary": ["false ceiling", "tiled false ceiling"], "secondary": ["metal grid", "calcium silicate", "aerolite"]},
}

# -- Keywords for comprehensive or technical descriptions, used for scoring --
COMPREHENSIVE_INDICATORS = ["including", "complete", "as per", "at all levels", "as directed", "all leads and lifts"]
TECHNICAL_INDICATORS = ["grade", "thickness", "size", "mm", "cum", "sqm", "kg", "is", "cement", "sand", "aggregate", "micron", "gauge", "dia"]

# ==============================================================================
# ===== SPECIFIC MATERIAL DEFINITIONS & MAPPING =====
# ==============================================================================

# -- Maps a material type from MATERIAL_INDICATORS to a final BOQ Category --
MATERIAL_CATEGORY_MAP = {
    "CONCRETE": "STRUCTURE_CONCRETE", "STEEL": "REINFORCEMENT_STEEL", "STEEL_STRUCTURAL": "STEEL_WORKS",
    "MASONRY": "MASONRY_WALL", "FLOORING": "FLOORING_TILE", "PLASTERING": "PLASTERING_WORK",
    "PAINTING": "PAINTING_FINISHING", "WATERPROOFING": "WATERPROOFING_MEMBRANE", "FORMWORK": "FORMWORK_SHUTTERING",
    "JOINERY": "JOINERY_DOORS", "DEMOLITION": "DEMOLITION", "EXCAVATION": "EXCAVATION_EARTHWORK",
    "GLAZING": "GLAZING", "MEP_ELECTRICAL": "MEP_ELECTRICAL",
    "ANTI-TERMITE": "WATERPROOFING_MEMBRANE", "POLYSHEET": "OTHERS", "DEWATERING": "OTHERS",
    "PIPES": "MEP_PLUMBING", "GROUT": "OTHERS", "COATING": "PAINTING_FINISHING",
    "FALSE_CEILING": "OTHERS",
}

# -- Defines the final, user-facing material name based on keywords --
#    (The order matters: more specific patterns should come first)
SPECIFIC_MATERIALS = [
    # MEP Electrical
    (["xlpe insulated aluminium conductor", "power cables"], "XLPE Aluminium Armoured Cable"),
    (["xlpe insulated copper unarmoured"], "XLPE Copper Unarmoured Cable"),
    (["cable end terminations"], "Cable Terminations"),
    (["mcb db", "distribution board"], "MCB Distribution Board"),
    (["pvc insulated copper conductor wires", "frls conduit"], "FRLS PVC Conduit & Copper Wires"),
    (["light points, fan points"], "Point Wiring"),
    (["tv, telephone & internet"], "Communication Conduit"),
    (["light fixtures"], "Light Fixtures"),
    (["ladder type cable tray"], "GI Ladder Cable Tray"),
    (["pipe electrode earth station"], "Pipe Electrode Earth Station"),
    (["gi earth strips"], "GI Earth Strips"),
    (["cable tray supports"], "MS Cable Tray Supports"),
    # Glazing
    (["aluminium glazed doors", "aluminium glazed windows"], "Aluminium Frame Glazing"),
    (["pvc glazed doors", "pvc glazed windows"], "PVC Frame Glazing"),
    # Earthwork
    (["earth work excavation", "excavation for foundation"], "Earthwork Excavation"),
    (["back filling", "backfilling"], "Earth Backfilling"),
    (["hard rock", "excavation in rock"], "Hard Rock Excavation"),
    # Concrete
    (["plain cement concrete m10", "pcc m10"], "PCC M10 Concrete"),
    (["m 25 grade concrete", "design mix concrete m 25"], "M25 Grade Concrete"),
    (["grade slab"], "Concrete Grade Slab"),
    (["pre cast rcc cover"], "Precast RCC Cover"),
    (["r.c.c. precast slab", "rcc precast slab"], "RCC Precast Slab"),
    (["saucer drain"], "Concrete Saucer Drain"),
    # Steel
    (["thermo mechanically treated (tmt)", "fe 550"], "TMT Steel (Fe 550)"),
    (["mechanical splicing", "coupler"], "Reinforcement Couplers"),
    # Formwork
    (["centering and shuttering", "system formworks"], "System Formwork"),
    # Flooring
    (["double charged vitrified tiles", "800 x 800"], "Vitrified Tiles (800x800)"),
    # Joinery
    (["fire rated steel door", "fire rated door"], "Fire Rated Steel Door"),
    (["ms door shutter", "crca sheet"], "MS Door Shutter"),
    # Waterproofing & Treatments
    (["anti termite treatment", "is 6315"], "Anti-Termite Treatment"),
    (["epoxy based grout"], "Epoxy Grout"),
    (["pu floor coating", "florthane"], "Polyurethane (PU) Floor Coating"),
    (["hdpe membrane"], "HDPE Waterproofing Membrane"),
    (["polyurethane coating"], "Polyurethane Waterproofing Coating"),
    (["geotextile membrane"], "Geotextile Membrane"),
    # Other Categories
    (["calcium silicate false ceiling", "aerolite"], "Calcium Silicate False Ceiling"),
    (["polythene paper", "polythene sheet"], "Polythene Sheet"),
    (["pumping out water", "dewatering"], "Dewatering Services"),
    # MEP Plumbing
    (["r.c.c np 2 class hume pipe", "hume pipe"], "RCC Hume Pipe (NP2)"),
    (["dwc hdpe pipe", "hdpe pipe"], "DWC HDPE Pipe (SN8)"),
    (["gi class \"c\" puddle flanged pipes", "gi pipe"], "GI Puddle Flanged Pipe (Class C)"),
    # Core Items
    (["demolition of rcc"], "Demolished RCC"), (["p.c.c"], "Plain Cement Concrete (PCC)"),
    (["rcc"], "Reinforced Cement Concrete (RCC)"), (["tmt bars"], "TMT Steel Bars"),
    (["brick masonry"], "Brick Masonry"), (["granite slab"], "Granite"), (["vitrified tiles"], "Vitrified Tiles"),
    (["mivan"], "Mivan Shuttering System"), (["teak wood frame"], "Teak Wood Door Frame"),
]

# -- Recognizes specialized materials and maps them to a category and final name --
SPECIALIZED_MATERIALS = {
    "chemical anchor": ("OTHERS", "Chemical Anchor Bolts"),
    "pvc pipe": ("MEP_PLUMBING", "PVC Pipes"),
    "conduit": ("MEP_ELECTRICAL", "Electrical Conduit"),
}

# -- Regex patterns to extract grade, ratio, or type from a description --
CONCRETE_GRADE_PATTERNS = [(r"\bm\s*(\d+(?:\.\d+)?)\b", "M{} Concrete")]
STEEL_GRADE_PATTERNS = [(r"\bfe\s*(\d+)\b", "Fe{} Steel")]
MORTAR_PATTERNS = [(r"cement mortar\s*(\d+\s*:\s*\d+)", "Cement Mortar {}")]

# ==============================================================================
# ===== CATEGORIZATION RULES & STRUCTURES =====
# ==============================================================================

# -- Defines how to extract structured data (material, grade, dimensions) for a given category --
STRUCTURED_MATERIAL_RULES = {
    "MEP_ELECTRICAL": {
        "material_types": [
            "XLPE Aluminium Armoured Cable", "XLPE Copper Unarmoured Cable", "Cable Terminations",
            "MCB Distribution Board", "FRLS PVC Conduit & Copper Wires", "Point Wiring", "Communication Conduit",
            "Light Fixtures", "Aviation Obstruction Light", "GI Ladder Cable Tray", "Pipe Electrode Earth Station",
            "GI Earth Strips", "Safety Signage (Shock Chart)", "MS Cable Tray Supports"
        ],
        "grade_pattern": r"(\d+\s*x\s*\d+\.?\d*)\s*sqmm", # Extracts cable size like "3 x 1.5sqmm"
        "grade_format": "{} sqmm",
        "dimension_pattern": r"(\d+)\s*mm\s*dia", # Extracts conduit size like "20 mm dia"
    },
    "STRUCTURE_CONCRETE": {"material_output": "Concrete", "grade_pattern": r"\bm\s*(\d+(?:\.\d+)?)\b", "grade_format": "M{}"},
    "REINFORCEMENT_STEEL": {"material_output": "Reinforcement Steel", "grade_pattern": r"\bfe\s*(\d+)\b", "grade_format": "Fe{}"},
    "STEEL_WORKS": {"material_output": "Structural Steel", "grade_format": "MS Angles, Plates, Sections"},
    "MASONRY_WALL": {"material_types": ["Brick Masonry", "Block Masonry"], "grade_pattern": r"(\d+\s*:\s*\d+)", "grade_format": "{}"},
    "PLASTERING_WORK": {"material_types": ["Cement Plaster", "Gypsum Plaster"], "thickness_pattern": r"(\d+)\s*mm"},
    "FLOORING_TILE": {"material_types": ["Granite", "Vitrified Tiles"], "thickness_pattern": r"(\d+)\s*mm"},
    "PAINTING_FINISHING": {"material_types": ["Paint", "Emulsion", "Polyurethane (PU) Floor Coating"]},
    "JOINERY_DOORS": {"material_types": ["Teak Wood Door Frame", "Fire Rated Steel Door"], "dimension_pattern": r"(\d+)\s*x\s*(\d+)\s*mm"},
    "GLAZING": {
        "material_types": ["Aluminium Frame Glazing", "PVC Frame Glazing"],
        "grade_pattern": r"(\d+)\s*mm\s*thick\s*.*?glass", # Extracts the glass thickness
        "grade_format": "{}mm Glass"
    },
    "MEP_PLUMBING": {"material_types": ["RCC Hume Pipe", "DWC HDPE Pipe", "GI Pipe"], "dimension_pattern": r"(\d+)\s*mm\s*dia"},
    "FALSE_CEILING": {"material_types": ["Calcium Silicate False Ceiling"], "dimension_pattern": r"(\d+)\s*x\s*(\d+)\s*mm"},
    "WATERPROOFING_MEMBRANE": {"material_types": ["HDPE Membrane", "Polyurethane Coating", "Geotextile Membrane"], "thickness_pattern": r"(\d+\.?\d*)\s*mm\s*thick"},
}

# ==============================================================================
# ===== LOCATION & FILTERING PATTERNS =====
# ==============================================================================

# -- Keywords to identify the physical location of the work --
LOCATION_MAPPING = {
    "Foundations (Substructure)": ["basement", "foundation", "footing", "pile", "raft", "substructure"],
    "Load-bearing Structural Frame": ["beam", "column", "slab", "structural", "frame", "rcc"],
    "Non-loadbearing Items": ["internal wall", "partition", "non structural", "masonry"],
    "Facades": ["external", "exterior", "facade", "cladding"],
    "Building Installations": ["electrical", "plumbing", "mep", "hvac"],
    "Other Building Elements": ["roofing", "flooring", "door", "window", "finishing", "ceiling"]
}

# -- Regex patterns to identify lines that are ONLY locations and should inherit context --
LOCATION_ONLY_PATTERNS = [
    r"^(tower\s+raft|in\s+walls?|platform\s+lvl|top\s+slab).*",
    r"^(\d+(st|nd|rd|th)\s+floor\s+lvl).*",
    r"^(ground\s+floor|first\s+floor|basement|roof|terrace).*",
    r"^(at\s+all\s+levels?).*",
    r"^(beams?|columns?|slabs?|walls?|stairs?)$",
]

# -- Regex for complex headers that combine multiple actions --
COMPLEX_HEADER_PATTERNS = [
    r"providing and fixing.*including.*as per",
    r"supplying.*receiving.*shifting.*stacking",
    r"providing.*fabricating.*erecting.*anchoring",
]

# -- Regex to REJECT a line from being a header --
REJECT_PATTERNS = [r"^basic price", r"^rate\s*:", r"^note\s*:?", r"support chairs"]
# -- Regex to identify a line as a NOTE --
NOTE_PATTERNS = [r"^basic price", r"^rate\s*:", r"^note\s*:?", r"^rate to include"]
# -- Phrases to exclude from being classified as a NOTE if they are part of a structural description --
STRUCTURAL_EXCLUSIONS = [r"providing and laying", r"supplying and fixing"]

# -- Keywords indicating that a material is incidental (e.g., an aggregate) and not the primary work --
INCIDENTAL_MATERIAL_PATTERNS = {
    "granite_aggregate": [r"using\s+graded\s+granite\s+aggregate", r"granite\s+aggregate", r"granite\s+chips"]
}

# ==============================================================================
# ===== ACTIVE HELPER FUNCTIONS (Used by ai_extractor4.py) =====
# ==============================================================================

def is_demolition_activity(description: str) -> bool:
    """Check if description is a demolition activity using core patterns."""
    desc_lower = description.lower()
    demolition_keywords = ["demolition", "demolishing", "dismantling", "breaking", "removal of existing"]
    return any(keyword in desc_lower for keyword in demolition_keywords)

def is_incidental_material_mention(description: str, material_type: str) -> bool:
    """Check if a material is mentioned as a secondary component (e.g., aggregate)."""
    desc_lower = description.lower()
    if material_type == "granite":
        for pattern in INCIDENTAL_MATERIAL_PATTERNS["granite_aggregate"]:
            if re.search(pattern, desc_lower):
                return True
    return False

def is_location_only_line(description: str) -> bool:
    """Check if a line only describes a location and should inherit header context."""
    desc_lower = description.lower().strip()
    for pattern in LOCATION_ONLY_PATTERNS:
        if re.match(pattern, desc_lower):
            return True
    return False

def get_activity_priority_category(description: str) -> tuple:
    """Determine category based on high-priority activities over incidental materials."""
    desc_lower = description.lower()

    if any(activity in desc_lower for activity in ["laying concrete", "placing concrete", "p.c.c", "pcc"]):
        if "granite aggregate" in desc_lower:
            return "STRUCTURE_CONCRETE", "Plain Cement Concrete (PCC)", 90.0
        return "STRUCTURE_CONCRETE", "Concrete", 85.0

    if any(activity in desc_lower for activity in ["formwork for", "shuttering for", "centering and shuttering"]):
        return "FORMWORK_SHUTTERING", "Formwork Material", 85.0

    if any(activity in desc_lower for activity in ["fabricating", "cutting steel", "bending steel", "fixing tmt"]):
        return "REINFORCEMENT_STEEL", "Reinforcement Steel", 85.0

    return None, None, 0.0

def check_specialized_material(description: str) -> tuple:
    """Check for specialized materials and return their predefined category and name."""
    desc_lower = description.lower()
    for material_key, (category, material_name) in SPECIALIZED_MATERIALS.items():
        if material_key in desc_lower:
            return category, material_name
    return None, None

def extract_header_specifications(text: str) -> dict:
    """Extract key specifications (grade, thickness, etc.) from a header text."""
    text_lower = text.lower()
    specs = {"grade": None, "thickness": None, "ratio": None}

    grade_match = re.search(r"\bm\s*(\d+(?:\.\d+)?)\b", text_lower)
    if grade_match:
        specs["grade"] = f"M{grade_match.group(1)}"

    thickness_match = re.search(r"(\d+)\s*mm\s*thick", text_lower)
    if thickness_match:
        specs["thickness"] = f"{thickness_match.group(1)}mm"

    ratio_match = re.search(r"(\d+\s*:\s*\d+)", text_lower)
    if ratio_match:
        specs["ratio"] = ratio_match.group(1)

    return specs

def is_complex_header(description: str) -> bool:
    """Check if a header matches complex, multi-action patterns."""
    text_lower = description.lower()
    return any(re.search(pattern, text_lower) for pattern in COMPLEX_HEADER_PATTERNS)


