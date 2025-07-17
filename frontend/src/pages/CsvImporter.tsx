import React from 'react';
import CSVImporter from '../components/CSVImporter';

const CsvImporterPage: React.FC = () => {
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', padding: '2rem 0' }}>
      <CSVImporter />
    </div>
  );
};

export default CsvImporterPage;
