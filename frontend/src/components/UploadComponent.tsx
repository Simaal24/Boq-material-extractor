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
    <div className="text-center max-w-2xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-2">Upload Your Bill of Quantities</h2>
      <p className="text-gray-500 mb-8">Select your Excel file to begin.</p>
      
      <label htmlFor="file-upload" className="cursor-pointer">
        <div className="p-12 border-2 border-dashed rounded-xl border-gray-300 hover:border-teal-400 transition-colors">
          <div className="flex flex-col items-center text-gray-600">
            <svg className="w-16 h-16 mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
            {file && <p className="font-semibold text-teal-700 mt-4">{file.name}</p>}
            <p className="text-sm text-gray-400 mt-2">XLS, XLSX files supported</p>
          </div>
        </div>
      </label>
      
      {error && <p className="text-red-500 mt-4">{error}</p>}
      
      <button 
        onClick={handleAnalyzeClick} 
        disabled={!file || loading} 
        className="mt-8 w-full sm:w-auto bg-teal-500 text-white font-bold py-3 px-12 rounded-lg hover:bg-teal-600 disabled:bg-gray-400 transition-all shadow-md"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
    </div>
  );
};

export default UploadComponent;