import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Trash2, AlertCircle, CheckCircle, Info, Settings, FolderOpen } from 'lucide-react';
import { ImportResult, MessageState, PathSettings } from '../types';
import './CsvImporter.css';

const CSVImporter: React.FC = () => {
  const baseUrl = import.meta.env.VITE_BACKEND_BASEURL;
  const [importPath, setImportPath] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadMethod, setUploadMethod] = useState<'file' | 'path'>('file');
  const [message, setMessage] = useState<MessageState | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  
  // Path configuration state
  const [showPathConfig, setShowPathConfig] = useState(false);
  const [customVideoDir, setCustomVideoDir] = useState('');
  const [customThumbnailDir, setCustomThumbnailDir] = useState('');
  const [savePaths, setSavePaths] = useState(false);
  const [pathSettings, setPathSettings] = useState<PathSettings | null>(null);

  // Load current path settings on component mount
  useEffect(() => {
    loadPathSettings();
  }, []);

  const loadPathSettings = async () => {
    try {
      const response = await axios.get(`${baseUrl}/learn/settings/paths`);
      const settings = response.data as PathSettings;
      setPathSettings(settings);
      setCustomVideoDir(settings.custom_video_dir || '');
      setCustomThumbnailDir(settings.custom_thumbnail_dir || '');
    } catch (error) {
      console.error('Failed to load path settings:', error);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setMessage({
        type: 'error',
        text: 'Please select a CSV file to upload'
      });
      return;
    }

    setIsImporting(true);
    setMessage({ type: 'info', text: 'Uploading and importing CSV file...' });

    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // Add custom path parameters if provided
    const params = new URLSearchParams();
    if (customVideoDir.trim()) {
      params.append('custom_video_dir', customVideoDir.trim());
    }
    if (customThumbnailDir.trim()) {
      params.append('custom_thumbnail_dir', customThumbnailDir.trim());
    }
    if (savePaths) {
      params.append('save_paths', 'true');
    }

    try {
      const url = `${baseUrl}/learn/upload-csv${params.toString() ? '?' + params.toString() : ''}`;
      const response = await axios.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = response.data as ImportResult;
      
      if (result.imported !== undefined) {
        setMessage({
          type: 'success',
          text: `Successfully imported ${result.imported} videos${result.failed ? `. ${result.failed} videos failed to import.` : ''}`,
          details: result.failed_videos
        });
        
        // Reload path settings if they were saved
        if (savePaths) {
          loadPathSettings();
        }
      } else {
        setMessage({
          type: 'success',
          text: result.message || 'Import completed successfully'
        });
      }
    } catch (error: any) {
      console.error('Upload error:', error);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.error || 
                          'Failed to upload and import CSV file. Please try again.';
      
      setMessage({
        type: 'error',
        text: errorMessage
      });
    } finally {
      setIsImporting(false);
    }
  };

  const handleImport = async () => {
    if (!importPath.trim()) {
      setMessage({
        type: 'error',
        text: 'Please enter a file path'
      });
      return;
    }

    setIsImporting(true);
    setMessage({ type: 'info', text: 'Importing videos from CSV...' });

    try {
      const response = await axios.post(
        `${baseUrl}/learn/import-csv?file_path=${encodeURIComponent(importPath)}`
      );

      const result = response.data as ImportResult;
      
      if (result.imported !== undefined) {
        setMessage({
          type: 'success',
          text: `Successfully imported ${result.imported} videos${result.failed ? `. ${result.failed} videos failed to import.` : ''}`,
          details: result.failed_videos
        });
      } else {
        setMessage({
          type: 'success',
          text: result.message || 'Import completed successfully'
        });
      }
    } catch (error: any) {
      console.error('Import error:', error);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.error || 
                          'Failed to import CSV file. Please check the file path and try again.';
      
      setMessage({
        type: 'error',
        text: errorMessage
      });
    } finally {
      setIsImporting(false);
    }
  };

  const handleDeleteAll = async () => {
    const confirmMessage = 
      'Are you sure you want to delete all videos from the database?\n\n' +
      'This action cannot be undone and will permanently remove all video records.';
    
    if (!window.confirm(confirmMessage)) {
      return;
    }

    setIsDeleting(true);
    setMessage({ type: 'info', text: 'Deleting all videos...' });

    try {
      const response = await axios.post(`${baseUrl}/learn/delete-all-videos`);
      
      setMessage({
        type: 'success',
        text: (response.data as any).message || 'All videos deleted successfully'
      });
      
      // Clear the import path since all data is now gone
      setImportPath('');
    } catch (error: any) {
      console.error('Delete error:', error);
      
      const errorMessage = error.response?.data?.detail || 
                          error.response?.data?.error || 
                          'Failed to delete videos. Please try again.';
      
      setMessage({
        type: 'error',
        text: errorMessage
      });
    } finally {
      setIsDeleting(false);
    }
  };

  const renderMessage = () => {
    if (!message) return null;

    const getIcon = () => {
      switch (message.type) {
        case 'success': return <CheckCircle size={20} />;
        case 'error': return <AlertCircle size={20} />;
        case 'info': return <Info size={20} />;
      }
    };

    return (
      <div className={`csv-importer__message csv-importer__message--${message.type}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {getIcon()}
          <span>{message.text}</span>
        </div>
        
        {message.details && message.details.length > 0 && (
          <div className="csv-importer__failed-videos">
            <h4>Failed Videos:</h4>
            <ul>
              {message.details.map((video, index) => (
                <li key={index}>{video}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="csv-importer">
      <div className="csv-importer__header">
        <h2 className="csv-importer__title">CSV Video Importer</h2>
        <p className="csv-importer__subtitle">
          Import ASL video metadata from a CSV file into the database
        </p>
      </div>

      {/* Path Configuration Section */}
      <div className="csv-importer__section">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
          <Settings size={20} />
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#374151', margin: 0 }}>
            Directory Configuration
          </h3>
          <button
            type="button"
            onClick={() => setShowPathConfig(!showPathConfig)}
            className="csv-importer__button csv-importer__button--secondary"
            style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
          >
            {showPathConfig ? 'Hide' : 'Show'}
          </button>
        </div>

        {pathSettings && (
          <div style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1rem' }}>
            <p><strong>Current Settings:</strong></p>
            <p>Video Directory: {pathSettings.custom_video_dir || pathSettings.default_video_dir}</p>
            <p>Thumbnail Directory: {pathSettings.custom_thumbnail_dir || pathSettings.default_thumbnail_dir}</p>
          </div>
        )}

        {showPathConfig && (
          <div style={{ backgroundColor: '#f9fafb', padding: '1rem', borderRadius: '8px', marginBottom: '1rem' }}>
            <div className="csv-importer__input-group">
              <label className="csv-importer__label">
                <FolderOpen size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
                Custom Video Directory (optional)
              </label>
              <input
                type="text"
                value={customVideoDir}
                onChange={(e) => setCustomVideoDir(e.target.value)}
                placeholder="Enter custom video directory path (e.g., E:/my_videos)"
                className="csv-importer__input"
                disabled={isImporting || isDeleting}
              />
            </div>

            <div className="csv-importer__input-group">
              <label className="csv-importer__label">
                <FolderOpen size={16} style={{ display: 'inline', marginRight: '0.5rem' }} />
                Custom Thumbnail Directory (optional)
              </label>
              <input
                type="text"
                value={customThumbnailDir}
                onChange={(e) => setCustomThumbnailDir(e.target.value)}
                placeholder="Enter custom thumbnail directory path"
                className="csv-importer__input"
                disabled={isImporting || isDeleting}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="checkbox"
                id="save-paths"
                checked={savePaths}
                onChange={(e) => setSavePaths(e.target.checked)}
                disabled={isImporting || isDeleting}
              />
              <label htmlFor="save-paths" style={{ fontSize: '0.875rem', color: '#374151' }}>
                Save these paths as default for future imports
              </label>
            </div>
          </div>
        )}
      </div>

      <div className="csv-importer__form">
        <div className="csv-importer__input-group">
          <label className="csv-importer__label">Import Method</label>
          <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
            <button
              type="button"
              onClick={() => setUploadMethod('file')}
              className={`csv-importer__button ${uploadMethod === 'file' ? 'csv-importer__button--primary' : 'csv-importer__button--secondary'}`}
              style={{ padding: '0.5rem 1rem' }}
            >
              Upload File
            </button>
            <button
              type="button"
              onClick={() => setUploadMethod('path')}
              className={`csv-importer__button ${uploadMethod === 'path' ? 'csv-importer__button--primary' : 'csv-importer__button--secondary'}`}
              style={{ padding: '0.5rem 1rem' }}
            >
              Server Path
            </button>
          </div>
        </div>

        {uploadMethod === 'file' ? (
          <div className="csv-importer__input-group">
            <label htmlFor="csv-file" className="csv-importer__label">
              Select CSV File
            </label>
            <input
              id="csv-file"
              type="file"
              accept=".csv,.tsv"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              className="csv-importer__input"
              disabled={isImporting || isDeleting}
            />
            {selectedFile && (
              <p style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.5rem' }}>
                Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </div>
        ) : (
          <div className="csv-importer__input-group">
            <label htmlFor="csv-path" className="csv-importer__label">
              CSV File Path (Server)
            </label>
            <input
              id="csv-path"
              type="text"
              value={importPath}
              onChange={(e) => setImportPath(e.target.value)}
              placeholder="Enter server file path (e.g., E:/asl_videos_nlp_enhanced_updated.csv)"
              className="csv-importer__input"
              disabled={isImporting || isDeleting}
            />
          </div>
        )}

        <div className="csv-importer__actions">
          <button
            onClick={uploadMethod === 'file' ? handleFileUpload : handleImport}
            disabled={
              isImporting || 
              isDeleting || 
              (uploadMethod === 'file' ? !selectedFile : !importPath.trim())
            }
            className="csv-importer__button csv-importer__button--primary"
          >
            <Upload size={18} />
            {isImporting ? 'Processing...' : uploadMethod === 'file' ? 'Upload & Import' : 'Import Videos'}
          </button>

          <button
            onClick={handleDeleteAll}
            disabled={isImporting || isDeleting}
            className="csv-importer__button csv-importer__button--danger"
          >
            <Trash2 size={18} />
            {isDeleting ? 'Deleting...' : 'Delete All Videos'}
          </button>
        </div>
      </div>

      {renderMessage()}

      <div className="csv-importer__info">
        <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '1rem', color: '#374151' }}>
          CSV Import Options & Requirements
        </h3>
        
        <div style={{ marginBottom: '1rem' }}>
          <h4 style={{ fontSize: '1rem', fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>
            Import Methods:
          </h4>
          <ul style={{ color: '#6b7280', lineHeight: '1.6', marginLeft: '1rem' }}>
            <li>• <strong>Upload File:</strong> Select and upload a CSV file directly from your computer</li>
            <li>• <strong>Server Path:</strong> Enter the path to a CSV file already on the server</li>
          </ul>
        </div>

        <div>
          <h4 style={{ fontSize: '1rem', fontWeight: '500', marginBottom: '0.5rem', color: '#4b5563' }}>
            CSV Requirements:
          </h4>
          <ul style={{ color: '#6b7280', lineHeight: '1.6', marginLeft: '1rem' }}>
            <li>• CSV file must contain a "videos" column with video filenames</li>
            <li>• Optional columns: word, title, description, Difficulty, Category</li>
            <li>• Video files must exist in the configured VIDEO_DIR on the server</li>
            <li>• Supported video formats: MP4, AVI, MOV, MKV</li>
            <li>• Thumbnails will be automatically generated and saved</li>
            <li>• Maximum file size: 100MB</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default CSVImporter;
