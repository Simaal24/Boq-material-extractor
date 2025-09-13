import React, { useState } from 'react';

interface Sheet {
  name: string;
  isBoq: boolean;
  headers?: string[];
  columnMappings?: any;
}

interface SheetSelectorProps {
  fileData: {
    fileIdentifier: string;
    sheets: Sheet[];
  };
  onExtract: (fileIdentifier: string, selectedSheets: string[]) => void;
  onBack: () => void;
}

const SheetSelectorComponent: React.FC<SheetSelectorProps> = ({ fileData, onExtract, onBack }) => {
  // Pre-select sheets where mandatory columns are successfully identified
  const [selectedSheets, setSelectedSheets] = useState<string[]>(
    fileData.sheets.filter(s => s.isBoq && s.columnMappings?.validation_status === "SUCCESS").map(s => s.name)
  );

  const handleCheckboxChange = (sheetName: string) => {
    setSelectedSheets(prev => 
      prev.includes(sheetName) 
        ? prev.filter(name => name !== sheetName)
        : [...prev, sheetName]
    );
  };

  return (
    <div className="max-w-2xl mx-auto w-full">
      <h2 className="text-2xl font-bold text-gray-800 mb-1 text-center">Select Worksheets</h2>
      <p className="text-gray-500 mb-4 text-center">Choose the sheets to extract data from.</p>
      
      <div className="border border-gray-200 rounded-lg bg-white shadow-sm divide-y divide-gray-200 max-h-96 overflow-y-auto">
        {fileData.sheets.map(sheet => {
          const hasValidColumns = sheet.columnMappings?.validation_status === "SUCCESS";
          const isPreselected = hasValidColumns && sheet.isBoq;
          
          return (
            <div key={sheet.name} className={`p-2 transition-colors ${
              isPreselected ? 'bg-teal-50' : 'hover:bg-gray-50'
            }`}>
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedSheets.includes(sheet.name)}
                  onChange={() => handleCheckboxChange(sheet.name)}
                  className="h-4 w-4 rounded border-gray-300 text-teal-600 focus:ring-teal-500"
                />
                <span className="ml-3 text-sm text-gray-700 font-medium">{sheet.name}</span>
                
                {/* Status badges */}
                {hasValidColumns ? (
                  <span className="ml-auto bg-green-100 text-green-800 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                    Headers Identified
                  </span>
                ) : sheet.isBoq ? (
                  <span className="ml-auto bg-red-100 text-red-800 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                    Headers Not Identified
                  </span>
                ) : (
                  <span className="ml-auto bg-gray-100 text-gray-600 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                    No BOQ Data
                  </span>
                )}
              </label>
              
              {/* Show identified headers when selected */}
              {selectedSheets.includes(sheet.name) && hasValidColumns && sheet.headers && sheet.headers.length > 0 && (
                <div className="mt-1 ml-7">
                  <p className="text-xs text-gray-500 mb-1 font-medium">Identified Headers:</p>
                  <div className="flex flex-wrap gap-1">
                    {sheet.headers.map(header => (
                      <span key={header} className="bg-green-200 text-green-800 text-xs font-medium px-1.5 py-0.5 rounded">
                        {header}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Show warning for unidentified sheets */}
              {selectedSheets.includes(sheet.name) && sheet.isBoq && !hasValidColumns && (
                <div className="mt-1 ml-7">
                  <p className="text-xs text-red-600 mb-1 font-medium">Header identification incomplete</p>
                  <p className="text-xs text-gray-500">May not extract properly due to missing headers.</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
      
      <div className="mt-8 flex justify-center space-x-4">
        <button 
          onClick={onBack}
          className="bg-gray-200 text-gray-700 font-bold py-3 px-8 rounded-lg hover:bg-gray-300 transition-all shadow-md"
        >
          Back
        </button>
        <button 
          onClick={() => onExtract(fileData.fileIdentifier, selectedSheets)} 
          disabled={selectedSheets.length === 0} 
          className="bg-teal-500 text-white font-bold py-3 px-12 rounded-lg hover:bg-teal-600 disabled:bg-gray-400 transition-all shadow-md"
        >
          Extract Data
        </button>
      </div>
    </div>
  );
};

export default SheetSelectorComponent;