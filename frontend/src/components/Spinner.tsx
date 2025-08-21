import React from 'react';

interface SpinnerProps {
  message?: string;
}

const Spinner: React.FC<SpinnerProps> = ({ message = "Processing your data..." }) => (
  <div className="flex flex-col items-center justify-center p-10">
    <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-teal-500"></div>
    <p className="text-lg text-gray-600 mt-4">{message}</p>
  </div>
);

export default Spinner;