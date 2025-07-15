import React, { useState } from 'react';
import axios from 'axios';

const CSVImporter: React.FC = () => {
  const baseUrl   = import.meta.env.VITE_BACKEND_BASEURL;
  const [importPath, setImportPath] = useState('');
  const [importMessage, setImportMessage] = useState('');

  const handleImport = async () => {
    if (!importPath) {
      setImportMessage('Please enter a file path');
      return;
    }
    try {
      const res: any = await axios.post(`${baseUrl}/learn/import-csv?file_path=${importPath}`);
      setImportMessage(res.data.message || res.data.error);
    } catch (error) {
      setImportMessage('Error importing file');
      console.error('Import error:', error);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm('Are you sure you want to delete all videos? This action cannot be undone.')) {
      return;
    }
    try {
      const res: any = await axios.post(`${baseUrl}/learn/delete-all-videos`);
      setImportMessage(res.data.message || 'All videos deleted successfully');
    } catch (error) {
      setImportMessage('Error deleting videos');
      console.error('Delete error:', error);
    }
  };

  return (
    <div style={{ margin: '20px 0', padding: '10px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h3>Import Local CSV via API</h3>
      <input 
        type="text" 
        value={importPath} 
        onChange={(e) => setImportPath(e.target.value)} 
        placeholder="Enter server file path (e.g., ./data/videos.csv)" 
        style={{ width: '100%', marginBottom: '10px' }}
      />
      <button onClick={handleImport} style={{ backgroundColor: '#007bff', color: 'white', padding: '8px 16px', borderRadius: '4px', marginRight: '10px' }}>Import</button>
      <button onClick={handleDeleteAll} style={{ backgroundColor: '#dc3545', color: 'white', padding: '8px 16px', borderRadius: '4px' }}>Delete All Videos</button>
      {importMessage && <p>{importMessage}</p>}
    </div>
  );
};

export default CSVImporter;
