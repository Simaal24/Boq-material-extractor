import React, { useState } from 'react';
import axios from 'axios';

interface UploadComponentProps {
  onUploadSuccess: (data: any) => void;
}

const UploadComponent: React.FC<UploadComponentProps> = ({ onUploadSuccess }) => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setError(''); // Clear any previous errors
    }
  };

  const handleAnalyzeClick = async () => {
    if (!file) return;
    
    setLoading(true);
    setError('');
    
    try {
      // Create FormData object
      const formData = new FormData();
      formData.append('file', file);
      
      // Make API call to backend
      const response = await axios.post('http://127.0.0.1:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Pass response data to parent component
      onUploadSuccess(response.data);
      
    } catch (err: any) {
      console.error('Upload error:', err);
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError('File analysis failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="text-center max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-1">Upload Your Bill of Quantities</h2>
      <p className="text-gray-500 mb-4">Select your Excel file to begin.</p>
      
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="p-8 border-2 border-dashed rounded-lg border-gray-300 hover:border-teal-400 transition-colors">
          <div className="flex flex-col items-center text-gray-600">
            <svg className="w-12 h-12 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
            <span className="bg-white rounded-md font-medium text-teal-600 hover:text-teal-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-teal-500">
              Click to upload a file
            </span>
            <input 
              id="file-upload" 
              name="file-upload" 
              type="file" 
              className="sr-only" 
              onChange={handleFileChange} 
              accept=".xls,.xlsx,.xlsm" 
            />
            {file && <p className="font-semibold text-teal-700 mt-2">{file.name}</p>}
            <p className="text-xs text-gray-400 mt-1">XLS, XLSX files supported</p>
          </div>
        </div>
      </label>
      
      <div className="mt-3 p-2 bg-yellow-50 border border-yellow-200 rounded-lg text-left text-xs text-yellow-800" role="alert">
        <div className="flex items-start">
          <svg className="w-4 h-4 mr-2 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd"></path>
          </svg>
          <div>
            <span className="font-semibold">For successful extraction, please ensure your file:</span>
            <ul className="list-disc list-inside mt-1 space-y-0.5">
              <li>Includes columns for <strong>Description</strong>, <strong>Unit</strong>, and <strong>Quantity</strong>.</li>
              <li>Clearly marks any total rows to avoid double-counting.</li>
            </ul>
          </div>
        </div>
      </div>
      
      {error && <p className="text-red-500 mt-3 text-sm">{error}</p>}
      
      <button 
        onClick={handleAnalyzeClick} 
        disabled={!file || loading} 
        className="mt-4 w-full sm:w-auto bg-teal-500 text-white font-bold py-2.5 px-10 rounded-lg hover:bg-teal-600 disabled:bg-gray-400 transition-all shadow-md"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
    </div>
  );
};

export default UploadComponent;