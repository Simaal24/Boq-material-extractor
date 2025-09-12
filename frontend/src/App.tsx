import React, { useState } from 'react';
import axios from 'axios';
import UploadComponent from './components/UploadComponent';
import SheetSelectorComponent from './components/SheetSelectorComponent';
import VerificationGridComponent from './components/VerificationGridComponent';
import SummaryComponent from './components/SummaryComponent';
import Spinner from './components/Spinner';

function App() {
  const [appState, setAppState] = useState('upload');
  const [fileData, setFileData] = useState<any>(null);
  const [extractionData, setExtractionData] = useState<any>(null);
  const [summaryData, setSummaryData] = useState<any>(null);

  const handleUploadSuccess = (data: any) => {
    console.log('Upload successful:', data);
    
    // Transform backend response to match frontend expectations
    const transformedData = {
      fileIdentifier: data?.analysis?.file_path || '',
      sheets: data?.analysis?.worksheets?.map((ws: any) => ({
        name: ws?.name || '',
        isBoq: ws?.column_mappings?.validation_status === "SUCCESS" || false,
        // Add identified headers based on column mappings
        headers: ws?.column_mappings && ws?.suspected_headers?.length > 0 ? 
          (() => {
            const headerRow = ws.suspected_headers[0]; // Get first (best) header row
            if (headerRow && headerRow.length > 2) { // [row_index, headers_array, confidence]
              const headerNames = headerRow[1]; // Headers array
              const mappings = ws.column_mappings;
              return [
                mappings.description_column !== undefined ? headerNames[mappings.description_column] : null,
                mappings.unit_column !== undefined ? headerNames[mappings.unit_column] : null,
                mappings.quantity_column !== undefined ? headerNames[mappings.quantity_column] : null
              ].filter(Boolean);
            }
            return [];
          })() : [],
        columnMappings: ws?.column_mappings || null
      })) || []
    };
    
    setFileData(transformedData);
    setAppState('select');
  };

  const handleExtract = async (fileIdentifier: string, selectedSheets: string[]) => {
    console.log('Extract called with:', fileIdentifier, selectedSheets);
    setAppState('processing');
    
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/extract', {
        file_path: fileIdentifier,
        selected_worksheets: selectedSheets
      });
      
      console.log('Extraction successful:', response.data);
      
      // PRESERVE WORKSHEET GROUPING - Keep items organized by sheet
      const sheets: any[] = [];
      const extractionData = response?.data?.extraction_data;
      if (extractionData && typeof extractionData === 'object') {
        Object.entries(extractionData).forEach(([sheetName, sheetData]: [string, any]) => {
          if (sheetData?.data && Array.isArray(sheetData.data)) {
            const items = sheetData.data.map((item: any, index: number) => ({
              id: `${sheetName}_${item?.original_row_index || index}`,
              Category: item?.final_category || 'OTHERS',
              Material: item?.final_material || 'Mixed Materials',
              Grade: item?.extracted_grade || '',
              Specifications: item?.technical_specs || item?.extracted_dimensions || '',
              Unit: item?.original_unit || 'Each', // Use original unit directly
              Quantity: item?.original_quantity || '', // Use original quantity directly
              originalRow: item?.original_description || '',
              // Store all original data for verification
              originalData: item?.original_data || {},
              rowIndex: item?.original_row_index || index,
              worksheetName: sheetName  // Add worksheet name
            }));
            
            if (items.length > 0) {
              sheets.push({
                name: sheetName,
                items: items
              });
            }
          }
        });
      }
      
      setExtractionData({ sheets });
      setAppState('verify');
      
    } catch (err: any) {
      console.error('Extraction error:', err);
      alert('Extraction failed: ' + (err?.response?.data?.error || 'Unknown error'));
      setAppState('select');
    }
  };

  const handleGenerateSummary = async (correctedData: any[]) => {
    console.log('Generate summary called with:', correctedData);
    setAppState('processing');
    
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/summarize', {
        verified_data: correctedData
      });
      
      console.log('Summary generated:', response.data);
      setSummaryData(response.data);
      setAppState('summary');
      
    } catch (err: any) {
      console.error('Summary error:', err);
      alert('Summary generation failed: ' + (err?.response?.data?.error || 'Unknown error'));
      setAppState('verify');
    }
  };

  const handleReset = () => {
    setAppState('upload');
    setFileData(null);
    setExtractionData(null);
    setSummaryData(null);
  };

  const handleBack = () => {
    switch (appState) {
      case 'select':
        setAppState('upload');
        setFileData(null);
        break;
      case 'verify':
        setAppState('select');
        setExtractionData(null);
        break;
      case 'summary':
        setAppState('verify');
        setSummaryData(null);
        break;
      default:
        break;
    }
  };

  const renderContent = () => {
    switch (appState) {
      case 'upload':
        return <UploadComponent onUploadSuccess={handleUploadSuccess} />;
      case 'select':
        return <SheetSelectorComponent fileData={fileData} onExtract={handleExtract} onBack={handleBack} />;
      case 'processing':
        return <Spinner message="Processing your request..." />;
      case 'verify':
        return <VerificationGridComponent extractionData={extractionData} onGenerateSummary={handleGenerateSummary} onBack={handleBack} />;
      case 'summary':
        return <SummaryComponent summaryData={summaryData} onReset={handleReset} onBack={handleBack} />;
      default:
        return <UploadComponent onUploadSuccess={handleUploadSuccess} />;
    }
  };
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
      <header className="bg-white shadow-sm w-full">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <div className="flex items-center">
            <svg className="w-8 h-8 text-teal-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V7a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            <h1 className="text-2xl font-bold text-gray-800">BOQ Analysis Tool</h1>
          </div>
          {appState !== 'upload' && (
            <button onClick={handleReset} className="text-sm text-gray-500 hover:text-gray-800 transition">
              Reset
            </button>
          )}
        </div>
      </header>
      
      <main className="flex-grow flex items-center justify-center p-4 w-full">
        <div className="w-full max-w-7xl mx-auto">
          {renderContent()}
        </div>
      </main>
      
      <footer className="text-center py-6 text-sm text-gray-500 w-full">
        <p>&copy; 2025 BOQ App. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;