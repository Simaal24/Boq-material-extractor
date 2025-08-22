"""
BOQ Summarizer Module

This module handles the final summarization of verified BOQ data.
It takes user-corrected data and aggregates it by Category, Material, Grade, and Specification.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class FixedBOQSummarizer:
    """Handles Material Summarization with aggregation by Category, Material, Grade, and Specification"""
    
    def __init__(self):
        # Category priority for sorting (logical construction sequence)
        self.category_priority = {
            "EXCAVATION_EARTHWORK": 1,
            "STRUCTURE_CONCRETE": 2,
            "REINFORCEMENT_STEEL": 3,
            "FORMWORK_SHUTTERING": 4,
            "MASONRY_WALL": 5,
            "PLASTERING_WORK": 6,
            "FLOORING_TILE": 7,
            "WATERPROOFING_MEMBRANE": 8,
            "PAINTING_FINISHING": 9,
            "JOINERY_DOORS": 10,
            "STEEL_WORKS": 11,
            "MEP_ELECTRICAL": 12,
            "MEP_PLUMBING": 13,
            "DEMOLITION": 14,
            "OTHERS": 15
        }
        
        # Location priority for sorting (construction sequence)
        self.location_priority = {
            "Foundations (Substructure)": 1,
            "Load-bearing Structural Frame": 2,
            "Non-loadbearing Items": 3,
            "Facades": 4,
            "Building Installations": 5,
            "Other Building Elements": 6
        }
        
        logger.info("📊 BOQ Summarizer initialized - Material Summary generation")

    def create_material_summary_sheet_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Material Summary - No Double Counting, All Rows Included"""
        logger.info("📊 Creating Material Summary - No double counting, all rows included")
        
        try:
            # Step 1: Ensure Original_Row exists for tracking
            if 'Original_Row' not in df.columns:
                if 'Original_Row_Ref' in df.columns:
                    df['Original_Row'] = df['Original_Row_Ref']
                    logger.info("📊 Using Original_Row_Ref as Original_Row")
                else:
                    df['Original_Row'] = df.index + 2  # Excel-like numbering (header + 0-based index)
                    logger.info("📊 Created Original_Row from index")
            
            total_input_rows = len(df)
            logger.info(f"📊 Total input rows: {total_input_rows}")
            
            # Step 2: Detect subtotal/total rows to avoid double counting
            desc_col = self.find_description_column(df)
            if not desc_col:
                logger.warning("📊 No description column found, using first text column")
                text_cols = [col for col in df.columns if df[col].dtype == 'object']
                desc_col = text_cols[0] if text_cols else None
            
            if desc_col:
                subtotal_patterns = [
                    r"sub\s*total", r"subtotal", r"total.*qty", r"grand\s*total", 
                    r"running\s*total", r"cumulative", r"sub-total", 
                    r"total\s+of", r"subtotal\s+of", r"overall\s+total",
                    r"total\s+conventional\s+concrete", r"total.*works?$"
                ]
                
                # Create case-insensitive regex pattern
                combined_pattern = '|'.join(f'({pattern})' for pattern in subtotal_patterns)
                df['_is_subtotal'] = df[desc_col].astype(str).str.contains(
                    combined_pattern, case=False, na=False, regex=True
                )
                subtotal_count = df['_is_subtotal'].sum()
                logger.info(f"📊 Detected {subtotal_count} subtotal/total rows to exclude")
            else:
                df['_is_subtotal'] = False
                logger.warning("📊 Could not detect subtotal patterns - no description column")
            
            # Step 3: Define inclusion criteria (broader than just LINE_ITEM)
            # Find quantity and unit columns
            qty_col = self.find_quantity_column(df)
            unit_col = self.find_unit_column(df)
            
            # Convert quantity to numeric for evaluation
            has_quantity = pd.Series(False, index=df.index)
            if qty_col and qty_col in df.columns:
                df_qty_numeric = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
                has_quantity = df_qty_numeric > 0
                logger.info(f"📊 Found {has_quantity.sum()} rows with quantity > 0")
            
            # Check for meaningful material/category data
            has_material_data = pd.Series(False, index=df.index)
            if 'Stage3_Final_Category' in df.columns:
                has_material_data = (
                    df['Stage3_Final_Category'].notna() & 
                    (df['Stage3_Final_Category'] != '') &
                    (df['Stage3_Final_Category'] != 'OTHERS')
                ) | (
                    df['Stage3_Final_Material'].notna() & 
                    (df['Stage3_Final_Material'] != '') &
                    (df['Stage3_Final_Material'] != 'Mixed Materials')
                )
                logger.info(f"📊 Found {has_material_data.sum()} rows with meaningful material data")
            
            # Check for unit and description
            has_unit_and_desc = pd.Series(False, index=df.index)
            if unit_col and desc_col:
                has_unit_and_desc = (
                    df[unit_col].notna() & 
                    (df[unit_col] != '') &
                    df[desc_col].notna() & 
                    (df[desc_col].astype(str).str.len() > 10)
                )
                logger.info(f"📊 Found {has_unit_and_desc.sum()} rows with unit and meaningful description")
            
            # Inclusion criteria: NOT subtotal AND (LINE_ITEM OR has_quantity OR has_material_data OR has_unit_and_desc)
            is_line_item = df.get('Item_Type', '') == 'LINE_ITEM'
            
            include_mask = (
                ~df['_is_subtotal'] & 
                (is_line_item | has_quantity | has_material_data | has_unit_and_desc)
            )
            
            included_df = df[include_mask].copy()
            logger.info(f"📊 Included {len(included_df)} rows after filtering (excluded {total_input_rows - len(included_df)} subtotal/irrelevant rows)")
            
            if included_df.empty:
                logger.warning("📊 No rows included after filtering")
                return pd.DataFrame({
                    'Message': ['No valid rows found after filtering'],
                    'Total_Input_Rows': [total_input_rows],
                    'Subtotal_Rows_Excluded': [subtotal_count],
                    'Final_Included_Rows': [0]
                })
            
            # Step 4: Ensure required columns exist with proper defaults
            required_cols = {
                'Stage3_Final_Category': 'OTHERS',
                'Stage3_Final_Material': 'Mixed Materials',
                'Stage3_Extracted_Grade': '',
                'Stage3_Extracted_Location': 'Other Building Elements'
            }
            
            for col, default_val in required_cols.items():
                if col not in included_df.columns:
                    included_df[col] = default_val
                    logger.info(f"📊 Created missing column {col} with default: {default_val}")
                else:
                    included_df[col] = included_df[col].fillna(default_val)
            
            # Add Stage3_Extracted_Specification column for better grouping
            if 'Stage3_Extracted_Specification' not in included_df.columns:
                # Try to extract specifications from technical specs or dimensions
                if 'Stage3_Technical_Specs' in included_df.columns:
                    included_df['Stage3_Extracted_Specification'] = included_df['Stage3_Technical_Specs'].fillna('')
                elif 'Stage3_Extracted_Dimensions' in included_df.columns:
                    included_df['Stage3_Extracted_Specification'] = included_df['Stage3_Extracted_Dimensions'].fillna('')
                else:
                    included_df['Stage3_Extracted_Specification'] = ''
                logger.info("📊 Created Stage3_Extracted_Specification column for grouping")
            else:
                included_df['Stage3_Extracted_Specification'] = included_df['Stage3_Extracted_Specification'].fillna('')
            
            # Step 5: Convert qty/amount to numeric
            if qty_col and qty_col in included_df.columns:
                included_df[qty_col] = pd.to_numeric(included_df[qty_col], errors='coerce').fillna(0)
                total_input_qty = included_df[qty_col].sum()
                logger.info(f"📊 Total quantity from included rows: {total_input_qty}")
            else:
                logger.warning("📊 No quantity column found")
                included_df['Quantity'] = 0
                qty_col = 'Quantity'
                total_input_qty = 0
            
            amount_col = self.find_amount_column(included_df)
            if amount_col and amount_col in included_df.columns:
                included_df[amount_col] = pd.to_numeric(included_df[amount_col], errors='coerce').fillna(0)
                total_input_amount = included_df[amount_col].sum()
                logger.info(f"📊 Total amount from included rows: {total_input_amount}")
            else:
                logger.warning("📊 No amount column found")
                total_input_amount = 0
            
            # Ensure unit column exists
            if not unit_col or unit_col not in included_df.columns:
                included_df['Unit'] = 'Each'
                unit_col = 'Unit'
                logger.info("📊 Created default Unit column")
            
            # Step 6: Group rows by category, material, grade, specification, unit
            group_columns = [
                'Stage3_Final_Category',
                'Stage3_Final_Material', 
                'Stage3_Extracted_Grade',
                'Stage3_Extracted_Specification',
                unit_col
            ]
            
            logger.info(f"📊 Grouping by columns: {group_columns}")
            
            # Prepare aggregation functions
            agg_functions = {
                'Original_Row': lambda x: self.format_source_rows(x.tolist())
            }
            
            if qty_col:
                agg_functions[qty_col] = 'sum'
            if amount_col and amount_col in included_df.columns:
                agg_functions[amount_col] = 'sum'
            
            # Perform aggregation
            logger.info("📊 Starting aggregation...")
            grouped = included_df.groupby(group_columns, as_index=False, dropna=False)
            summary_df = grouped.agg(agg_functions)
            
            # Step 7: Add Line_Items_Count
            count_df = included_df.groupby(group_columns, dropna=False).size().reset_index(name='Line_Items_Count')
            summary_df = summary_df.merge(count_df, on=group_columns, how='left')
            
            logger.info(f"📊 Created {len(summary_df)} material groups from {len(included_df)} included rows")
            
            # Step 8: Rename columns and reorder
            column_renames = {
                'Stage3_Final_Category': 'Category',
                'Stage3_Final_Material': 'Material',
                'Stage3_Extracted_Grade': 'Grade',
                'Stage3_Extracted_Specification': 'Specification',
                unit_col: 'Unit',
                'Original_Row': 'Source_Row_Numbers'
            }
            
            if qty_col in summary_df.columns:
                column_renames[qty_col] = 'OVERALL QTY'
            if amount_col and amount_col in summary_df.columns:
                column_renames[amount_col] = 'TOTAL AMOUNT'
            
            # Apply renames only for existing columns
            existing_renames = {old: new for old, new in column_renames.items() if old in summary_df.columns}
            summary_df = summary_df.rename(columns=existing_renames)
            
            # Define column order
            desired_columns = ['Category', 'Material', 'Specification', 'Grade', 'Unit']
            if 'OVERALL QTY' in summary_df.columns:
                desired_columns.append('OVERALL QTY')
            if 'TOTAL AMOUNT' in summary_df.columns:
                desired_columns.append('TOTAL AMOUNT')
            desired_columns.extend(['Line_Items_Count', 'Source_Row_Numbers'])
            
            # Select available columns
            available_columns = [col for col in desired_columns if col in summary_df.columns]
            summary_df = summary_df[available_columns]
            
            # Step 9: Sort by category priority
            summary_df['_Sort_Priority'] = summary_df['Category'].map(self.category_priority).fillna(99)
            summary_df = summary_df.sort_values(['_Sort_Priority', 'Material', 'Grade', 'Specification', 'Unit'])
            summary_df = summary_df.drop(columns=['_Sort_Priority'])
            summary_df = summary_df.reset_index(drop=True)
            
            # Add Summary_ID
            summary_df.insert(0, 'Summary_ID', range(1, len(summary_df) + 1))
            
            # Step 10: Create totals row
            totals_data = {
                'Summary_ID': 'TOTAL',
                'Category': 'ALL CATEGORIES',
                'Material': 'PROJECT TOTAL',
                'Specification': '',
                'Grade': '',
                'Unit': 'MIXED',
                'Line_Items_Count': summary_df['Line_Items_Count'].sum()
            }
            
            if 'OVERALL QTY' in summary_df.columns:
                summary_total_qty = summary_df['OVERALL QTY'].sum()
                totals_data['OVERALL QTY'] = summary_total_qty
                logger.info(f"📊 Summary total quantity: {summary_total_qty} (Input total: {total_input_qty})")
                
                # Check for quantity mismatch
                if abs(summary_total_qty - total_input_qty) > 0.01:
                    logger.warning(f"📊 QUANTITY MISMATCH: Summary={summary_total_qty}, Input={total_input_qty}")
                else:
                    logger.info("📊 ✅ Quantity totals match!")
            
            if 'TOTAL AMOUNT' in summary_df.columns:
                summary_total_amount = summary_df['TOTAL AMOUNT'].sum()
                totals_data['TOTAL AMOUNT'] = summary_total_amount
                logger.info(f"📊 Summary total amount: {summary_total_amount} (Input total: {total_input_amount})")
            
            if 'Source_Row_Numbers' in summary_df.columns:
                totals_data['Source_Row_Numbers'] = f"All {len(summary_df)} groups"
            
            # Add totals row
            totals_row = pd.DataFrame([totals_data])
            summary_df = pd.concat([summary_df, totals_row], ignore_index=True)
            
            # Step 11: Debug check for missing rows
            try:
                included_original_rows = set(included_df['Original_Row'].dropna())
                
                # Extract all original rows from Source_Row_Numbers
                summary_original_rows = set()
                for source_rows in summary_df['Source_Row_Numbers'].dropna():
                    if source_rows and isinstance(source_rows, str) and source_rows != f"All {len(summary_df)} groups":
                        # Parse row numbers from formatted string
                        numbers = re.findall(r'\d+', source_rows)
                        summary_original_rows.update(int(num) for num in numbers)
                
                missing_rows = included_original_rows - summary_original_rows
                if missing_rows:
                    logger.warning(f"📊 MISSING ROWS: {len(missing_rows)} rows not found in summary: {sorted(list(missing_rows))[:10]}...")
                else:
                    logger.info("📊 ✅ All included rows are tracked in summary!")
                    
            except Exception as debug_error:
                logger.warning(f"📊 Could not perform debug check: {debug_error}")
            
            logger.info(f"📊 ✅ Material Summary: {len(summary_df)-1} groups + totals")
            return summary_df
            
        except Exception as e:
            logger.error(f"📊 CRITICAL ERROR in material summary: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return pd.DataFrame({
                'Error_Message': ['Material Summary creation failed'],
                'Error_Details': [str(e)],
                'Input_Rows': [len(df) if 'df' in locals() else 0],
                'Pandas_Version': [pd.__version__]
            })

    def find_description_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find description column with flexible naming"""
        desc_candidates = [
            'description', 'item description', 'work description', 'particulars',
            'item', 'work', 'details', 'desc', 'specification'
        ]
        
        # Check exact matches first
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in desc_candidates:
                logger.info(f"📊 Found description column (exact): {col}")
                return col
        
        # Check partial matches
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(candidate in col_lower for candidate in desc_candidates):
                logger.info(f"📊 Found description column (partial): {col}")
                return col
        
        # Find longest text column as fallback
        text_columns = [col for col in df.columns if df[col].dtype == 'object']
        if text_columns:
            text_lengths = {col: df[col].astype(str).str.len().mean() for col in text_columns}
            longest_col = max(text_lengths, key=text_lengths.get)
            logger.info(f"📊 Using longest text column as description: {longest_col}")
            return longest_col
        
        logger.warning("📊 No description column found")
        return None

    def find_unit_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find unit column with flexible naming"""
        unit_candidates = ['unit', 'units', 'uom', 'u.o.m', 'measure', 'measurement', 'u/m', 'original_unit']
        
        # Check exact matches first
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in unit_candidates:
                logger.info(f"📊 Found unit column (exact): {col}")
                return col
        
        # Check partial matches
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(candidate in col_lower for candidate in unit_candidates):
                logger.info(f"📊 Found unit column (partial): {col}")
                return col
        
        logger.warning("📊 No unit column found")
        return None

    def find_quantity_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find quantity column with flexible naming"""
        qty_candidates = [
            'quantity', 'qty', 'quan', 'no', 'nos', 'number', 'count',
            'overall qty', 'total qty', 'total quantity'
        ]
        
        # Check exact matches first
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in qty_candidates:
                logger.info(f"📊 Found quantity column (exact): {col}")
                return col
        
        # Check partial matches (avoid amount/cost columns)
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for candidate in qty_candidates:
                if candidate in col_lower and 'amount' not in col_lower and 'cost' not in col_lower:
                    logger.info(f"📊 Found quantity column (partial): {col}")
                    return col
        
        logger.warning("📊 No quantity column found")
        return None

    def find_amount_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find amount/cost column with flexible naming"""
        amount_candidates = [
            'amount', 'cost', 'value', 'price', 'total', 'sum', 'rs', 'inr',
            'total amount', 'total cost', 'total value', 'overall amount'
        ]
        
        # Check exact matches first
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in amount_candidates:
                logger.info(f"📊 Found amount column (exact): {col}")
                return col
        
        # Check partial matches (avoid quantity columns)
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for candidate in amount_candidates:
                if candidate in col_lower and 'qty' not in col_lower and 'quantity' not in col_lower:
                    logger.info(f"📊 Found amount column (partial): {col}")
                    return col
        
        logger.warning("📊 No amount column found")
        return None

    def format_source_rows(self, row_list: List) -> str:
        """Format original source row references for traceability"""
        # Remove any non-numeric or invalid entries
        valid_rows = []
        for row in row_list:
            try:
                if pd.notna(row) and row != '':
                    valid_rows.append(int(row))
            except (ValueError, TypeError):
                continue
        
        if not valid_rows:
            return "N/A"
        
        # Sort and format
        valid_rows = sorted(set(valid_rows))  # Remove duplicates and sort
        
        # If too many rows, show range
        if len(valid_rows) > 15:
            return f"Rows {valid_rows[0]}-{valid_rows[-1]} ({len(valid_rows)} items)"
        else:
            return f"Rows {', '.join(map(str, valid_rows))}"

    def create_simplified_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a simplified summary with only: Category, Material, Unit, Grade, Quantity
        """
        logger.info(f"📊 SIMPLIFIED SUMMARY: Starting with {len(df)} rows")
        logger.info(f"📊 Input DataFrame columns: {list(df.columns)}")
        logger.info(f"📊 Sample input data: {df.head(1).to_dict('records') if not df.empty else 'Empty DataFrame'}")
        
        # Map frontend column names to our processing (try multiple variations)
        required_columns = {
            'Category': 'Category',
            'Material': 'Material', 
            'Unit': 'Unit',
            'Grade': 'Grade',
            'Quantity': 'Quantity'
        }
        
        # Alternative column name mappings
        column_alternatives = {
            'Category': ['Category', 'category', 'CATEGORY', 'final_category', 'Stage3_Final_Category'],
            'Material': ['Material', 'material', 'MATERIAL', 'final_material', 'Stage3_Final_Material'],
            'Unit': ['Unit', 'unit', 'UNIT', 'original_unit', 'Original_Unit'],
            'Grade': ['Grade', 'grade', 'GRADE', 'extracted_grade', 'Stage3_Extracted_Grade'],
            'Quantity': ['Quantity', 'quantity', 'QUANTITY', 'original_quantity', 'Original_Quantity']
        }
        
        # Check which columns exist (try alternatives)
        available_cols = {}
        for frontend_col, expected_col in required_columns.items():
            found_col = None
            
            # Try the expected column first
            if expected_col in df.columns:
                found_col = expected_col
            else:
                # Try alternatives
                for alt_col in column_alternatives[frontend_col]:
                    if alt_col in df.columns:
                        found_col = alt_col
                        break
            
            if found_col:
                available_cols[frontend_col] = found_col
                logger.info(f"✅ Found column: {found_col} (for {frontend_col})")
            else:
                logger.warning(f"❌ Missing column: {frontend_col} (tried: {column_alternatives[frontend_col]})")
        
        # If we don't have the basic columns, return empty summary
        if 'Category' not in available_cols or 'Material' not in available_cols:
            logger.error("❌ Cannot create summary - missing Category or Material columns")
            return pd.DataFrame(columns=['Category', 'Material', 'Unit', 'Grade', 'Quantity'])
        
        # Select and clean the data
        working_df = df.copy()
        
        # Ensure required columns exist with defaults
        if 'Unit' not in available_cols:
            working_df['Unit'] = 'Each'
            available_cols['Unit'] = 'Unit'
            
        if 'Grade' not in available_cols:
            working_df['Grade'] = ''
            available_cols['Grade'] = 'Grade'
            
        if 'Quantity' not in available_cols:
            working_df['Quantity'] = 0
            available_cols['Quantity'] = 'Quantity'
        
        # Convert quantity to numeric
        working_df[available_cols['Quantity']] = pd.to_numeric(
            working_df[available_cols['Quantity']], errors='coerce'
        ).fillna(0)
        
        # Clean empty/null values
        for col in ['Category', 'Material', 'Unit', 'Grade']:
            if col in available_cols:
                working_df[available_cols[col]] = working_df[available_cols[col]].fillna('').astype(str)
        
        # Group by Category, Material, Unit, Grade and sum quantities
        group_cols = [
            available_cols['Category'],
            available_cols['Material'], 
            available_cols['Unit'],
            available_cols['Grade']
        ]
        
        logger.info(f"📊 Grouping by: {group_cols}")
        
        summary_df = working_df.groupby(group_cols, as_index=False, dropna=False).agg({
            available_cols['Quantity']: 'sum'
        })
        
        # Rename columns to standard names
        summary_df = summary_df.rename(columns={
            available_cols['Category']: 'Category',
            available_cols['Material']: 'Material',
            available_cols['Unit']: 'Unit', 
            available_cols['Grade']: 'Grade',
            available_cols['Quantity']: 'Quantity'
        })
        
        # Sort by category priority, then by material
        def get_category_priority(category):
            return self.category_priority.get(category, 999)
        
        summary_df['_sort_priority'] = summary_df['Category'].apply(get_category_priority)
        summary_df = summary_df.sort_values(['_sort_priority', 'Material', 'Grade']).drop('_sort_priority', axis=1)
        
        # Filter out zero quantities
        summary_df = summary_df[summary_df['Quantity'] > 0]
        
        logger.info(f"📊 SIMPLIFIED SUMMARY: Created {len(summary_df)} material groups")
        logger.info(f"📊 Total Quantity: {summary_df['Quantity'].sum()}")
        
        return summary_df


def generate_summary_dataframe(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of corrected data and generates a simplified material summary.
    
    Summarizes by: Category, Material, Unit, Grade, Quantity (no specifications or amounts)

    Args:
        data_df: The user-verified BOQ data.

    Returns:
        A pandas DataFrame containing the simplified material summary.
    """
    summarizer = FixedBOQSummarizer()
    # Use the new simplified method
    summary_df = summarizer.create_simplified_summary(data_df)
    return summary_df