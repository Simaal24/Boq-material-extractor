import React, { useState } from 'react';

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

interface VerificationGridProps {
  extractionData: {
    items: ExtractedItem[];
  };
  onGenerateSummary: (correctedData: ExtractedItem[]) => void;
}

const VerificationGridComponent: React.FC<VerificationGridProps> = ({ extractionData, onGenerateSummary }) => {
  const [rowData, setRowData] = useState<ExtractedItem[]>(extractionData?.items || []);

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

  const materialOptions = [
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

  const handleCellChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>, rowIndex: number, field: keyof ExtractedItem) => {
    const newData = [...rowData];
    const value = e.target.value;
    newData[rowIndex] = {
      ...newData[rowIndex],
      [field]: value
    };
    setRowData(newData);
  };

  if (!extractionData?.items || extractionData.items.length === 0) {
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
      
      <div className="overflow-x-auto bg-white rounded-lg shadow">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
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
            {rowData.map((row, rowIndex) => (
              <tr key={row.id} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-4 py-4 whitespace-nowrap">
                  <select 
                    value={row.Category} 
                    onChange={(e) => handleCellChange(e, rowIndex, 'Category')} 
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
                    onChange={(e) => handleCellChange(e, rowIndex, 'Material')} 
                    className="w-full p-1 border border-gray-300 rounded bg-white text-sm focus:ring-teal-500 focus:border-teal-500"
                  >
                    {materialOptions.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <input 
                    type="text" 
                    value={row.Grade} 
                    onChange={(e) => handleCellChange(e, rowIndex, 'Grade')} 
                    className="w-full p-1 border border-gray-300 rounded text-sm focus:ring-teal-500 focus:border-teal-500"
                    placeholder="e.g., M25, Fe500"
                  />
                </td>
                <td className="px-4 py-4 whitespace-nowrap">
                  <input 
                    type="text" 
                    value={row.Specifications} 
                    onChange={(e) => handleCellChange(e, rowIndex, 'Specifications')} 
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
                    onChange={(e) => handleCellChange(e, rowIndex, 'Quantity')} 
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
          </tbody>
        </table>
      </div>
      
      <div className="mt-8 flex justify-center space-x-4">
        <button 
          onClick={() => onGenerateSummary(rowData)} 
          className="bg-teal-500 text-white font-bold py-3 px-8 rounded-lg hover:bg-teal-600 transition-all shadow-md"
        >
          Generate Summary
        </button>
        <button className="bg-gray-200 text-gray-700 font-bold py-3 px-8 rounded-lg hover:bg-gray-300 transition-all">
          Download Report
        </button>
      </div>
    </div>
  );
};

export default VerificationGridComponent;