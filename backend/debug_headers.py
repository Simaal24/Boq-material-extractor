#!/usr/bin/env python3
"""
Debug script to test header pattern matching with the actual headers from the user's file
"""

import re

# The actual headers from the user's file
test_headers = [
    "S.NO",
    "DESCRIPTION", 
    "GRADE OF CONC",
    "CEFF",
    "UNIT",
    "OVERALL QTY",
    "NET RATE",
    "TOTAL AMOUNT",
    "Estimation Remarks to Project team",
    "Project head Remarks"
]

# BOQ patterns from the pre_processor
boq_patterns = {
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
        r'(?i)(grade|grade\s*of\s*conc|concrete\s*grade|grade\s*of\s*concrete)',
        r'(?i)(class|type|category)',
    ],
    'cement': [
        r'(?i)(cement\s*co-?eff|cement\s*coefficient|cement|ceff|c\s*eff)',
    ],
    'remarks': [
        r'(?i)(remarks|notes|comments|observations)',
        r'(?i)(specification|remark)',
    ]
}

print("Testing BOQ Header Pattern Matching")
print("=" * 50)

matched_patterns = []
total_score = 0

for header in test_headers:
    header_matches = []
    print(f"\nTesting header: '{header}'")
    
    for pattern_name, patterns in boq_patterns.items():
        for pattern in patterns:
            if re.search(pattern, header):
                header_matches.append(pattern_name)
                matched_patterns.append(pattern_name)
                total_score += 1
                print(f"  -> Matches '{pattern_name}' pattern: {pattern}")
                break  # Only count first match per pattern type
    
    if not header_matches:
        print(f"  -> No pattern matches")

print(f"\nSUMMARY:")
print(f"Total pattern matches: {total_score}")
print(f"Unique patterns matched: {set(matched_patterns)}")
print(f"Headers processed: {len(test_headers)}")

# Calculate confidence like the actual code
non_empty_cells = len(test_headers)
if non_empty_cells > 0 and total_score > 0:
    confidence = min(total_score / non_empty_cells, 0.8)
    
    # Apply boosts
    unique_patterns = len(set(matched_patterns))
    if unique_patterns >= 4:
        confidence *= 1.2
    elif unique_patterns >= 3:
        confidence *= 1.1
    elif unique_patterns >= 2:
        confidence *= 1.05
    
    # Boost for critical patterns
    critical_patterns = ['sl_no', 'description', 'unit', 'quantity', 'rate', 'amount']
    if any(pattern in matched_patterns for pattern in critical_patterns):
        confidence *= 1.1
    
    # Key BOQ indicators boost
    key_boq_indicators = ['sl_no', 'description', 'quantity', 'unit', 'rate', 'amount']
    key_matches = sum(1 for pattern in matched_patterns if pattern in key_boq_indicators)
    
    if key_matches >= 3:
        confidence *= 1.5
        print(f"STRONG BOQ HEADER detected with {key_matches} key indicators")
    
    confidence = min(confidence, 0.9)
    
    print(f"Final confidence score: {confidence:.3f}")
    print(f"Key BOQ matches: {key_matches}/6")
    print(f"Would be accepted: {'YES' if confidence > 0.20 else 'NO'}")
else:
    print(f"No matches found - confidence would be 0")