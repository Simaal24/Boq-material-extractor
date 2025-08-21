import React, { useState } from 'react';

interface Sheet {
  name: string;
  isBoq: boolean;
}

interface SheetSelectorProps {
  fileData: {
    fileIdentifier: string;
    sheets: Sheet[];
  };
  onExtract: (fileIdentifier: string, selectedSheets: string[]) => void;
}

const SheetSelectorComponent: React.FC<SheetSelectorProps> = ({ fileData, onExtract }) => {
  // Pre-select sheets where isBoq is true
  const [selectedSheets, setSelectedSheets] = useState<string[]>(
    fileData.sheets.filter(s => s.isBoq).map(s => s.name)
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
      <h2 className="text-3xl font-bold text-gray-800 mb-2 text-center">Select Worksheets</h2>
      <p className="text-gray-500 mb-8 text-center">Choose the sheets to extract data from.</p>
      
      <div className="space-y-4 border border-gray-200 rounded-lg p-6 bg-white">
        {fileData.sheets.map(sheet => (
          <label key={sheet.name} className="flex items-center p-4 rounded-lg hover:bg-gray-50 cursor-pointer">
            <input 
              type="checkbox" 
              checked={selectedSheets.includes(sheet.name)} 
              onChange={() => handleCheckboxChange(sheet.name)} 
              className="h-5 w-5 rounded border-gray-300 text-teal-600 focus:ring-teal-500" 
            />
            <span className="ml-4 text-lg text-gray-700 font-medium">{sheet.name}</span>
            {sheet.isBoq && (
              <span className="ml-auto bg-teal-100 text-teal-800 text-xs font-semibold px-2.5 py-0.5 rounded-full">
                Recommended
              </span>
            )}
          </label>
        ))}
      </div>
      
      <div className="mt-8 text-center">
        <button 
          onClick={() => onExtract(fileData.fileIdentifier, selectedSheets)} 
          disabled={selectedSheets.length === 0} 
          className="w-full sm:w-auto bg-teal-500 text-white font-bold py-3 px-12 rounded-lg hover:bg-teal-600 disabled:bg-gray-400 transition-all shadow-md"
        >
          Extract Data
        </button>
      </div>
    </div>
  );
};

export default SheetSelectorComponent;