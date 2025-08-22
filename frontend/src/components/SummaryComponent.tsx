import React from 'react';

interface SummaryItem {
  Category: string;
  Material: string;
  Grade: string;
  Unit: string;
  Quantity: number;
}

interface SummaryComponentProps {
  summaryData: {
    summary?: SummaryItem[];
    summary_data?: any[];
    timestamp?: string;
    summary_groups?: number;
    total_quantity?: number;
    input_rows?: number;
  };
  onReset: () => void;
}

const SummaryComponent: React.FC<SummaryComponentProps> = ({ summaryData, onReset }) => {
  // Handle case where summaryData might not have the expected structure
  const summaryItems = summaryData?.summary || summaryData?.summary_data || [];
  const totalQuantity = summaryData?.total_quantity || 0;
  const summaryGroups = summaryData?.summary_groups || summaryItems.length;

  return (
    <div className="text-center max-w-6xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-4">Summary Generated Successfully!</h2>
      
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-teal-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-teal-800">Material Groups</h3>
            <p className="text-2xl font-bold text-teal-600">{summaryGroups}</p>
          </div>
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-blue-800">Total Quantity</h3>
            <p className="text-2xl font-bold text-blue-600">{totalQuantity.toFixed(2)}</p>
          </div>
        </div>
      </div>

      {summaryItems.length > 0 ? (
        <div className="overflow-x-auto bg-white rounded-lg shadow">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Material</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unit</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Grade</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {summaryItems.map((row, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{row.Category}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{row.Material}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{row.Unit}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{row.Grade}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right font-medium">
                    {typeof row.Quantity === 'number' ? row.Quantity.toFixed(2) : row.Quantity || 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow p-8">
          <p className="text-gray-500">No summary data available.</p>
        </div>
      )}

      <div className="mt-8 flex justify-center space-x-4">
        <button 
          onClick={onReset} 
          className="bg-teal-500 hover:bg-teal-600 text-white font-bold py-3 px-8 rounded-lg transition-all shadow-md"
        >
          Start Over
        </button>
        <button className="bg-gray-200 text-gray-700 font-bold py-3 px-8 rounded-lg hover:bg-gray-300 transition-all">
          Download Excel
        </button>
      </div>
    </div>
  );
};

export default SummaryComponent;