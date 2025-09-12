import React, { useState } from 'react';
import * as XLSX from 'xlsx';

interface ExtractedItem {
  id: number;
  Category: string;
  Material: string;
  Grade: string;
  Specifications: string;
  Unit: string;
  Quantity: string;
  originalRow: string;
  originalData: any;
  rowIndex: number;
}

interface SheetData {
  name: string;
  items: ExtractedItem[];
}

interface VerificationGridProps {
  extractionData: {
    sheets?: SheetData[];
    items?: ExtractedItem[]; // Keep for backward compatibility
  };
  onGenerateSummary: (correctedData: ExtractedItem[]) => void;
  onBack: () => void;
}

const VerificationGridComponent: React.FC<VerificationGridProps> = ({ extractionData, onGenerateSummary, onBack }) => {
  // Handle both sheets-based and legacy items-based data structure
  const initialData = extractionData?.sheets 
    ? extractionData.sheets.flatMap(sheet => sheet.items)  // Flatten sheets into items
    : extractionData?.items || [];
  
  const [rowData, setRowData] = useState<ExtractedItem[]>(initialData);
  const [sheetsData, setSheetsData] = useState<SheetData[]>(extractionData?.sheets || []);

  // Predefined options for dropdowns
  const categoryOptions = [
    'STRUCTURE_CONCRETE',
    'REINFORCEMENT_STEEL', 
    'STEEL_WORKS',
    'MASONRY_WALL',
    'PLASTERING_WORK',
    'FLOORING_TILE',
    'PAINTING_FINISHING',
    'FORMWORK_SHUTTERING',
    'EXCAVATION_EARTHWORK',
    'WATERPROOFING_MEMBRANE',
    'JOINERY_DOORS',
    'GLAZING',
    'MEP_ELECTRICAL',
    'MEP_PLUMBING',
    'DEMOLITION',
    'OTHERS'
  ];

  const baseMaterialOptions = [
    'Concrete',
    'Reinforcement Bars',
    'Structural Steel',
    'Brick Masonry',
    'Solid Block',
    'Concrete Block',
    'Cement Plastering',
    'Gypsum Plastering',
    'Vitrified Tiles',
    'Granite',
    'Ceramic Tiles',
    'Acrylic Emulsion Paint',
    'Primer',
    'Excavated Soil',
    'Earth/Soil',
    'Formwork Material',
    'HDPE Waterproofing Membrane',
    'Teak Wood Door Frame',
    'Fire Rated Steel Door',
    'Aluminium Frame Glazing',
    'XLPE Aluminium Armoured Cable',
    'MCB Distribution Board',
    'RCC Hume Pipe',
    'DWC HDPE Pipe',
    'Mixed Materials'
  ];

  // Dynamically include AI-extracted materials that aren't in the predefined list
  const getAllMaterialOptions = () => {
    const extractedMaterials = rowData.map(item => item.Material).filter(Boolean);
    const uniqueExtractedMaterials = Array.from(new Set(extractedMaterials));
    const newMaterials = uniqueExtractedMaterials.filter(material => 
      !baseMaterialOptions.includes(material)
    );
    
    // Combine base options with AI-extracted materials (put AI-extracted first for visibility)
    return [
      ...newMaterials.sort(),
      ...baseMaterialOptions
    ];
  };

  const materialOptions = getAllMaterialOptions();

  const handleCellChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>, itemId: string, field: keyof ExtractedItem) => {
    const value = e.target.value;
    
    // Update sheets data structure
    const newSheetsData = sheetsData.map(sheet => ({
      ...sheet,
      items: sheet.items.map(item => 
        item.id === itemId ? { ...item, [field]: value } : item
      )
    }));
    
    // Update flattened row data
    const newRowData = rowData.map(item => 
      item.id === itemId ? { ...item, [field]: value } : item
    );
    
    setSheetsData(newSheetsData);
    setRowData(newRowData);
  };

  // Check if a material was AI-extracted (not in predefined list)
  const isAIExtractedMaterial = (material: string) => {
    return material && !baseMaterialOptions.includes(material);
  };

  const downloadExcel = () => {
    const exportData = rowData.map(item => ({
      Category: item.Category,
      Material: item.Material,
      Grade: item.Grade,
      Specifications: item.Specifications,
      Unit: item.Unit,
      Quantity: item.Quantity,
      'Original Row': item.originalRow
    }));

    const ws = XLSX.utils.json_to_sheet(exportData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Verification Data');
    
    // Auto-size columns
    const colWidths = Object.keys(exportData[0] || {}).map(key => ({
      wch: Math.max(key.length, ...exportData.map(row => String(row[key] || '').length))
    }));
    ws['!cols'] = colWidths;
    
    const fileName = `verification_data_${new Date().toISOString().slice(0, 10)}.xlsx`;
    XLSX.writeFile(wb, fileName);
  };

  // Check for empty data in both data structures
  const hasData = (extractionData?.sheets && extractionData.sheets.length > 0) || 
                  (extractionData?.items && extractionData.items.length > 0);
                  
  if (!hasData) {
    return (
      <div className="text-center max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold text-gray-800 mb-4">No Data Extracted</h2>
        <p className="text-gray-500">No items were found in the selected worksheets.</p>
      </div>
    );
  }

  return (
    <div className="w-full">
      <h2 className="text-3xl font-bold text-gray-800 mb-2 text-center">Verify Extracted Data</h2>
      <p className="text-gray-500 mb-8 text-center">Review and correct the extracted items below.</p>
      
      <div className="overflow-x-auto bg-white rounded-lg shadow" style={{maxHeight: '60vh'}}>
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Material</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Grade</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Specifications</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unit</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Original Row</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sheetsData.length > 0 ? (
              // Show data grouped by worksheets with separators
              sheetsData.map(sheet => (
                <React.Fragment key={sheet.name}>
                  <tr>
                    <td colSpan={7} className="px-6 py-3 bg-teal-100 text-teal-800 font-bold text-sm">
                      Worksheet: {sheet.name}
                    </td>
                  </tr>
                  {sheet.items.map((row, rowIndex) => (
                    <tr key={row.id} className="hover:bg-gray-50">
                      <td className="px-4 py-4 whitespace-nowrap">
                        <select 
                          value={row.Category} 
                          onChange={(e) => handleCellChange(e, row.id, 'Category')} 
                          className="w-full p-1 border border-gray-300 rounded bg-white text-sm focus:ring-teal-500 focus:border-teal-500"
                        >
                          {categoryOptions.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </select>
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap">
                        <select 
                          value={row.Material} 
                          onChange={(e) => handleCellChange(e, row.id, 'Material')} 
                          className={`w-full p-1 border border-gray-300 rounded bg-white text-sm focus:ring-teal-500 focus:border-teal-500 ${
                            isAIExtractedMaterial(row.Material) ? 'bg-blue-50 border-blue-300' : ''
                          }`}
                          title={isAIExtractedMaterial(row.Material) ? 'AI-extracted material (not in predefined list)' : 'Predefined material option'}
                        >
                          {materialOptions.map(opt => (
                            <option 
                              key={opt} 
                              value={opt}
                              className={isAIExtractedMaterial(opt) ? 'bg-blue-50 font-semibold' : ''}
                            >
                              {isAIExtractedMaterial(opt) ? `🤖 ${opt}` : opt}
                            </option>
                          ))}
                        </select>
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap">
                        <input 
                          type="text" 
                          value={row.Grade} 
                          onChange={(e) => handleCellChange(e, row.id, 'Grade')} 
                          className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500"
                          placeholder="e.g., M25, Fe500"
                        />
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap">
                        <input 
                          type="text" 
                          value={row.Specifications} 
                          onChange={(e) => handleCellChange(e, row.id, 'Specifications')} 
                          className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500"
                          placeholder="Technical specifications"
                        />
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-700 bg-gray-50">
                        <span className="font-mono" title="Original unit from BOQ file">{row.Unit}</span>
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap">
                        <input 
                          type="text" 
                          value={row.Quantity} 
                          onChange={(e) => handleCellChange(e, row.id, 'Quantity')} 
                          className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500 font-mono"
                          placeholder="0"
                          title="Original quantity from BOQ file"
                        />
                      </td>
                      <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500 max-w-xs truncate" title={row.originalRow}>
                        {row.originalRow}
                      </td>
                    </tr>
                  ))}
                </React.Fragment>
              ))
            ) : (
              // Fallback to legacy flat structure
              rowData.map((row, rowIndex) => (
                <tr key={row.id} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <select 
                      value={row.Category} 
                      onChange={(e) => handleCellChange(e, row.id, 'Category')} 
                      className="w-full p-1 border border-gray-300 rounded bg-white text-sm focus:ring-teal-500 focus:border-teal-500"
                    >
                      {categoryOptions.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <select 
                      value={row.Material} 
                      onChange={(e) => handleCellChange(e, row.id, 'Material')} 
                      className={`w-full p-1 border border-gray-300 rounded bg-white text-sm focus:ring-teal-500 focus:border-teal-500 ${
                        isAIExtractedMaterial(row.Material) ? 'bg-blue-50 border-blue-300' : ''
                      }`}
                      title={isAIExtractedMaterial(row.Material) ? 'AI-extracted material (not in predefined list)' : 'Predefined material option'}
                    >
                      {materialOptions.map(opt => (
                        <option 
                          key={opt} 
                          value={opt}
                          className={isAIExtractedMaterial(opt) ? 'bg-blue-50 font-semibold' : ''}
                        >
                          {isAIExtractedMaterial(opt) ? `🤖 ${opt}` : opt}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <input 
                      type="text" 
                      value={row.Grade} 
                      onChange={(e) => handleCellChange(e, row.id, 'Grade')} 
                      className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500"
                      placeholder="e.g., M25, Fe500"
                    />
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <input 
                      type="text" 
                      value={row.Specifications} 
                      onChange={(e) => handleCellChange(e, row.id, 'Specifications')} 
                      className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500"
                      placeholder="Technical specifications"
                    />
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-700 bg-gray-50">
                    <span className="font-mono" title="Original unit from BOQ file">{row.Unit}</span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <input 
                      type="text" 
                      value={row.Quantity} 
                      onChange={(e) => handleCellChange(e, row.id, 'Quantity')} 
                      className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500 font-mono"
                      placeholder="0"
                      title="Original quantity from BOQ file"
                    />
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-500 max-w-xs truncate" title={row.originalRow}>
                    {row.originalRow}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      
      <div className="mt-8 flex justify-center space-x-4">
        <button 
          onClick={onBack}
          className="bg-gray-200 text-gray-700 font-bold py-3 px-8 rounded-lg hover:bg-gray-300 transition-all shadow-md"
        >
          Back
        </button>
        <button 
          onClick={downloadExcel}
          className="bg-blue-500 text-white font-bold py-3 px-8 rounded-lg hover:bg-blue-600 transition-all shadow-md"
        >
          Download Excel
        </button>
        <button 
          onClick={() => onGenerateSummary(rowData)} 
          className="bg-teal-500 text-white font-bold py-3 px-8 rounded-lg hover:bg-teal-600 transition-all shadow-md"
        >
          Generate Summary
        </button>
      </div>
    </div>
  );
};

export default VerificationGridComponent;