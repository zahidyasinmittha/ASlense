import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  Edit2, 
  Trash2, 
  Search,
  Eye,
  EyeOff,
  Save,
  X,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';
import { pslAPI } from '../services/api';

interface PSLAlphabetEntry {
  id: number;
  letter: string;
  file_path: string;
  label: string;
  difficulty: string;
  description?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

interface CreatePSLEntry {
  letter: string;
  file_path: string;
  label: string;
  difficulty: string;
  description?: string;
  is_active: boolean;
}

const PSLAdminPage: React.FC = () => {
  const [entries, setEntries] = useState<PSLAlphabetEntry[]>([]);
  const [filteredEntries, setFilteredEntries] = useState<PSLAlphabetEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [difficultyFilter, setDifficultyFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [showModal, setShowModal] = useState(false);
  const [editingEntry, setEditingEntry] = useState<PSLAlphabetEntry | null>(null);
  const [notification, setNotification] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const [formData, setFormData] = useState<CreatePSLEntry>({
    letter: '',
    file_path: '',
    label: '',
    difficulty: 'easy',
    description: '',
    is_active: true
  });

  // Fetch PSL alphabet entries
  const fetchEntries = async () => {
    try {
      setLoading(true);
      const response = await pslAPI.getAllAdmin();
      const data = response.data as PSLAlphabetEntry[];
      setEntries(data);
      setFilteredEntries(data);
    } catch (error) {
      showNotification('error', 'Failed to fetch PSL alphabet entries');
    } finally {
      setLoading(false);
    }
  };

  // Filter entries based on search and filters
  useEffect(() => {
    let filtered = entries;

    if (searchTerm) {
      filtered = filtered.filter(entry => 
        entry.letter.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
        entry.file_path.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (difficultyFilter) {
      filtered = filtered.filter(entry => entry.difficulty === difficultyFilter);
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(entry => 
        statusFilter === 'active' ? entry.is_active : !entry.is_active
      );
    }

    setFilteredEntries(filtered);
  }, [entries, searchTerm, difficultyFilter, statusFilter]);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  const handleCreate = async () => {
    try {
      await pslAPI.create(formData);
      showNotification('success', 'PSL alphabet entry created successfully');
      setShowModal(false);
      resetForm();
      fetchEntries();
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to create entry';
      showNotification('error', errorMessage);
    }
  };

  const handleUpdate = async () => {
    if (!editingEntry) return;

    try {
      await pslAPI.update(editingEntry.id, formData);
      showNotification('success', 'PSL alphabet entry updated successfully');
      setShowModal(false);
      setEditingEntry(null);
      resetForm();
      fetchEntries();
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to update entry';
      showNotification('error', errorMessage);
    }
  };

  const handleDelete = async (id: number, letter: string) => {
    if (!confirm(`Are you sure you want to delete the PSL entry for letter "${letter}"?`)) {
      return;
    }

    try {
      await pslAPI.delete(id);
      showNotification('success', 'PSL alphabet entry deleted successfully');
      fetchEntries();
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to delete entry';
      showNotification('error', errorMessage);
    }
  };

  const toggleStatus = async (id: number) => {
    try {
      await pslAPI.toggleStatus(id);
      showNotification('success', 'Entry status updated successfully');
      fetchEntries();
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 'Failed to update entry status';
      showNotification('error', errorMessage);
    }
  };

  const handleEdit = (entry: PSLAlphabetEntry) => {
    setEditingEntry(entry);
    setFormData({
      letter: entry.letter,
      file_path: entry.file_path,
      label: entry.label,
      difficulty: entry.difficulty,
      description: entry.description || '',
      is_active: entry.is_active
    });
    setShowModal(true);
  };

  const resetForm = () => {
    setFormData({
      letter: '',
      file_path: '',
      label: '',
      difficulty: 'easy',
      description: '',
      is_active: true
    });
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'hard': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  useEffect(() => {
    fetchEntries();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg ${
          notification.type === 'success' ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
        }`}>
          <div className="flex items-center space-x-2">
            {notification.type === 'success' ? <CheckCircle className="h-5 w-5" /> : <AlertTriangle className="h-5 w-5" />}
            <span>{notification.message}</span>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">PSL Alphabet Management</h1>
          <p className="text-gray-600">Manage Pakistan Sign Language alphabet entries and media files</p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
            {/* Search and Filters */}
            <div className="flex flex-col sm:flex-row gap-4 flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                <input
                  type="text"
                  placeholder="Search letters, labels, or files..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent w-full sm:w-64"
                />
              </div>

              <select
                value={difficultyFilter}
                onChange={(e) => setDifficultyFilter(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">All Difficulties</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={() => {
                  resetForm();
                  setEditingEntry(null);
                  setShowModal(true);
                }}
                className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus className="h-4 w-4" />
                <span>Add Entry</span>
              </button>
            </div>
          </div>
        </div>

        {/* Entries Table */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">
              PSL Alphabet Entries ({filteredEntries.length})
            </h2>
          </div>

          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-gray-600">Loading entries...</p>
            </div>
          ) : filteredEntries.length === 0 ? (
            <div className="p-8 text-center">
              <p className="text-gray-600">No PSL alphabet entries found.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Letter
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Label
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      File Path
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Difficulty
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredEntries.map((entry) => (
                    <tr key={entry.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-lg font-bold text-gray-900">{entry.letter}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{entry.label}</div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="text-sm text-gray-900 max-w-xs truncate" title={entry.file_path}>
                          {entry.file_path}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getDifficultyColor(entry.difficulty)}`}>
                          {entry.difficulty}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button
                          onClick={() => toggleStatus(entry.id)}
                          className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-semibold ${
                            entry.is_active
                              ? 'bg-green-100 text-green-800 hover:bg-green-200'
                              : 'bg-red-100 text-red-800 hover:bg-red-200'
                          }`}
                        >
                          {entry.is_active ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                          <span>{entry.is_active ? 'Active' : 'Inactive'}</span>
                        </button>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(entry.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex justify-end space-x-2">
                          <button
                            onClick={() => handleEdit(entry)}
                            className="text-blue-600 hover:text-blue-900 p-1 rounded hover:bg-blue-50"
                            title="Edit entry"
                          >
                            <Edit2 className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(entry.id, entry.letter)}
                            className="text-red-600 hover:text-red-900 p-1 rounded hover:bg-red-50"
                            title="Delete entry"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Add/Edit Modal */}
        {showModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-md w-full p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">
                  {editingEntry ? 'Edit PSL Entry' : 'Add New PSL Entry'}
                </h3>
                <button
                  onClick={() => {
                    setShowModal(false);
                    setEditingEntry(null);
                    resetForm();
                  }}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Letter
                  </label>
                  <input
                    type="text"
                    value={formData.letter}
                    onChange={(e) => setFormData({...formData, letter: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Alif"
                    maxLength={50}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Label
                  </label>
                  <input
                    type="text"
                    value={formData.label}
                    onChange={(e) => setFormData({...formData, label: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="1-A"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    File Path
                  </label>
                  <input
                    type="text"
                    value={formData.file_path}
                    onChange={(e) => setFormData({...formData, file_path: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="s0251-10hay.jpg"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Difficulty
                  </label>
                  <select
                    value={formData.difficulty}
                    onChange={(e) => setFormData({...formData, difficulty: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="easy">Easy</option>
                    <option value="medium">Medium</option>
                    <option value="hard">Hard</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description (Optional)
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({...formData, description: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows={3}
                    placeholder="How to sign this letter..."
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="is_active"
                    checked={formData.is_active}
                    onChange={(e) => setFormData({...formData, is_active: e.target.checked})}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <label htmlFor="is_active" className="ml-2 block text-sm text-gray-700">
                    Active
                  </label>
                </div>
              </div>

              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={() => {
                    setShowModal(false);
                    setEditingEntry(null);
                    resetForm();
                  }}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={editingEntry ? handleUpdate : handleCreate}
                  className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Save className="h-4 w-4" />
                  <span>{editingEntry ? 'Update' : 'Create'}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PSLAdminPage;
