import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { adminAPI } from '../services/api';

interface AdminStats {
  totalUsers: number;
  totalPredictions: number;
  averageAccuracy: number;
  activeUsers: number;
  totalVideos: number;
}

interface User {
  id: number;
  username: string;
  email: string;
  role: 'admin' | 'user';
  full_name?: string;
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

interface CreateUserData {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  role: 'admin' | 'user';
}

interface EditUserData {
  username?: string;
  email?: string;
  full_name?: string;
  role?: 'admin' | 'user';
  is_active?: boolean;
}

interface Video {
  id: number;
  title: string;
  word: string;
  category: string;
  difficulty: string;
  filePath: string;
  thumbnailPath?: string;
  duration?: number;
  createdAt: string;
}

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

interface SystemMetrics {
  predictionAccuracy: number;
  averageSessionTime: number;
  dailyActiveUsers: number;
  totalSessions: number;
}

interface ContactMessage {
  id: number;
  name: string;
  email: string;
  subject: string;
  message: string;
  status: 'unread' | 'read' | 'replied';
  created_at: string;
  updated_at: string;
  admin_notes?: string;
}

interface ContactStats {
  total: number;
  unread: number;
  read: number;
  replied: number;
}

const AdminDashboard: React.FC = () => {
  const { token, isAdmin, user } = useAuth();
  const navigate = useNavigate();
  
  // Debug: Log current user info
  console.log('AdminDashboard - Current user:', user);
  console.log('AdminDashboard - Is admin:', isAdmin);
  console.log('AdminDashboard - Token exists:', !!token);
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'videos' | 'psl' | 'metrics' | 'tools' | 'contact'>('overview');
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [pslEntries, setPslEntries] = useState<PSLAlphabetEntry[]>([]);
  const [filteredPslEntries, setFilteredPslEntries] = useState<PSLAlphabetEntry[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [contactMessages, setContactMessages] = useState<ContactMessage[]>([]);
  const [contactStats, setContactStats] = useState<ContactStats | null>(null);
  const [selectedMessage, setSelectedMessage] = useState<ContactMessage | null>(null);
  const [messageStatusFilter, setMessageStatusFilter] = useState<string>('all');
  const [showMessageModal, setShowMessageModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [pslLoading, setPslLoading] = useState(false);
  const [contactLoading, setContactLoading] = useState(false);
  const [showCreateUserModal, setShowCreateUserModal] = useState(false);
  const [showEditUserModal, setShowEditUserModal] = useState(false);
  const [showPslModal, setShowPslModal] = useState(false);
  const [editingPslEntry, setEditingPslEntry] = useState<PSLAlphabetEntry | null>(null);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [createUserData, setCreateUserData] = useState<CreateUserData>({
    username: '',
    email: '',
    password: '',
    full_name: '',
    role: 'user'
  });
  const [editUserData, setEditUserData] = useState<EditUserData>({});
  const [pslFormData, setPslFormData] = useState<CreatePSLEntry>({
    letter: '',
    file_path: '',
    label: '',
    difficulty: 'easy',
    description: '',
    is_active: true
  });
  const [userSearchTerm, setUserSearchTerm] = useState('');
  const [pslSearchTerm, setPslSearchTerm] = useState('');
  const [pslDifficultyFilter, setPslDifficultyFilter] = useState('');
  const [pslStatusFilter, setPslStatusFilter] = useState('all');
  const [pslPage, setPslPage] = useState(1);
  const [pslHasMore, setPslHasMore] = useState(true);
  const [pslLoadingMore, setPslLoadingMore] = useState(false);
  const [userRoleFilter, setUserRoleFilter] = useState<'all' | 'admin' | 'user'>('all');
  const [userStatusFilter, setUserStatusFilter] = useState<'all' | 'active' | 'inactive'>('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [usersPerPage] = useState(10);
  const [notification, setNotification] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  useEffect(() => {
    if (!isAdmin) {
      console.warn('User is not admin, redirecting to dashboard');
      navigate('/dashboard');
      return;
    }
    fetchAdminData();
  }, [isAdmin, navigate, token]);

  const fetchAdminData = async () => {
    try {
      // Fetch admin stats
      try {
        const statsResponse = await adminAPI.getStats();
        setStats(statsResponse.data as AdminStats);
      } catch (error) {
        console.error('Error fetching stats:', error);
      }

      // Fetch users
      try {
        const usersResponse = await adminAPI.getUsers();
        console.log('Fetching users:', usersResponse);
        setUsers(usersResponse.data as User[]);
      } catch (error) {
        console.error('Error fetching users:', error);
      }

      // Fetch videos
      try {
        const videosResponse = await adminAPI.getVideos();
        setVideos(videosResponse.data as Video[]);
      } catch (error) {
        console.error('Error fetching videos:', error);
      }

      // Fetch system metrics
      try {
        const metricsResponse = await adminAPI.getMetrics();
        setMetrics(metricsResponse.data as SystemMetrics);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }

      // Fetch contact messages and stats
      try {
        const contactResponse = await fetch('/api/v1/contact/admin/messages', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (contactResponse.ok) {
          const contactData = await contactResponse.json();
          setContactMessages(contactData);
        }

        const contactStatsResponse = await fetch('/api/v1/contact/admin/stats', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (contactStatsResponse.ok) {
          const contactStatsData = await contactStatsResponse.json();
          setContactStats(contactStatsData);
        }
      } catch (error) {
        console.error('Error fetching contact data:', error);
      }
    } catch (error) {
      console.error('Error fetching admin data:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleUserStatus = async (userId: number, currentStatus: boolean) => {
    try {
      const response = await adminAPI.toggleUserStatus(userId);
      
      setUsers(users.map(u => 
        u.id === userId ? { ...u, is_active: !currentStatus } : u
      ));
      const result = response.data as { message: string };
      showNotification('success', result.message);
    } catch (error: any) {
      console.error('Error toggling user status:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to toggle user status';
      showNotification('error', errorMessage);
    }
  };

  const createUser = async () => {
    try {
      if (!createUserData.username || !createUserData.email || !createUserData.password) {
        showNotification('error', 'Please fill in all required fields');
        return;
      }

      const response = await adminAPI.createUser(createUserData);
      const newUser = response.data as User;
      
      setUsers([...users, newUser]);
      setShowCreateUserModal(false);
      setCreateUserData({
        username: '',
        email: '',
        password: '',
        full_name: '',
        role: 'user'
      });
      showNotification('success', `User ${newUser.username} created successfully`);
    } catch (error: any) {
      console.error('Error creating user:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to create user';
      showNotification('error', errorMessage);
    }
  };

  const updateUser = async () => {
    if (!selectedUser) return;

    try {
      const response = await adminAPI.updateUser(selectedUser.id, editUserData);
      const updatedUser = response.data as User;
      
      setUsers(users.map(u => u.id === selectedUser.id ? updatedUser : u));
      setShowEditUserModal(false);
      setSelectedUser(null);
      setEditUserData({});
      showNotification('success', `User ${updatedUser.username} updated successfully`);
    } catch (error: any) {
      console.error('Error updating user:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to update user';
      showNotification('error', errorMessage);
    }
  };

  const deleteUser = async (userId: number) => {
    const user = users.find(u => u.id === userId);
    if (!user || !confirm(`Are you sure you want to deactivate ${user.username}?`)) return;

    try {
      const response = await adminAPI.deleteUser(userId);
      const result = response.data as { message: string };
      
      setUsers(users.map(u => u.id === userId ? { ...u, is_active: false } : u));
      showNotification('success', result.message);
    } catch (error: any) {
      console.error('Error deleting user:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to delete user';
      showNotification('error', errorMessage);
    }
  };

  const openEditModal = (user: User) => {
    setSelectedUser(user);
    setEditUserData({
      username: user.username,
      email: user.email,
      full_name: user.full_name || '',
      role: user.role,
      is_active: user.is_active
    });
    setShowEditUserModal(true);
  };

  // Contact Message Functions
  const fetchContactMessages = async (status?: string) => {
    try {
      setContactLoading(true);
      let url = '/api/v1/contact/admin/messages';
      if (status && status !== 'all') {
        url += `?status=${status}`;
      }
      
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        setContactMessages(data);
      }
    } catch (error) {
      console.error('Error fetching contact messages:', error);
      showNotification('error', 'Failed to fetch contact messages');
    } finally {
      setContactLoading(false);
    }
  };

  const markMessageAsRead = async (messageId: number) => {
    try {
      const response = await fetch(`/api/v1/contact/admin/messages/${messageId}/mark-read`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (response.ok) {
        setContactMessages(contactMessages.map(msg => 
          msg.id === messageId ? { ...msg, status: 'read' } : msg
        ));
        showNotification('success', 'Message marked as read');
      }
    } catch (error) {
      console.error('Error marking message as read:', error);
      showNotification('error', 'Failed to mark message as read');
    }
  };

  const markMessageAsReplied = async (messageId: number, adminNotes: string = '') => {
    try {
      const response = await fetch(`/api/v1/contact/admin/messages/${messageId}/mark-replied`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ admin_notes: adminNotes }),
      });
      
      if (response.ok) {
        setContactMessages(contactMessages.map(msg => 
          msg.id === messageId ? { ...msg, status: 'replied', admin_notes: adminNotes } : msg
        ));
        showNotification('success', 'Message marked as replied');
      }
    } catch (error) {
      console.error('Error marking message as replied:', error);
      showNotification('error', 'Failed to mark message as replied');
    }
  };

  const deleteContactMessage = async (messageId: number) => {
    if (!confirm('Are you sure you want to delete this contact message?')) return;
    
    try {
      const response = await fetch(`/api/v1/contact/admin/messages/${messageId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });
      
      if (response.ok) {
        setContactMessages(contactMessages.filter(msg => msg.id !== messageId));
        showNotification('success', 'Contact message deleted');
      }
    } catch (error) {
      console.error('Error deleting contact message:', error);
      showNotification('error', 'Failed to delete contact message');
    }
  };

  const openMessageModal = (message: ContactMessage) => {
    setSelectedMessage(message);
    setShowMessageModal(true);
    
    // Mark as read if unread
    if (message.status === 'unread') {
      markMessageAsRead(message.id);
    }
  };

  // PSL Management Functions
  const fetchPslEntries = async (page: number = 1, append: boolean = false) => {
    try {
      if (page === 1) {
        setPslLoading(true);
      } else {
        setPslLoadingMore(true);
      }
      
      const params = new URLSearchParams({
        skip: ((page - 1) * 20).toString(),
        limit: '20'
      });
      
      // Add search parameter if exists
      if (pslSearchTerm) {
        params.append('search', pslSearchTerm);
      }
      
      // Add filter parameters if exists
      if (pslDifficultyFilter) {
        params.append('difficulty', pslDifficultyFilter);
      }
      
      if (pslStatusFilter !== 'all') {
        params.append('is_active', pslStatusFilter === 'active' ? 'true' : 'false');
      }
      
      const response = await fetch(`/api/v1/psl-alphabet/admin/all?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        
        if (append && page > 1) {
          // Append new entries for infinite scroll
          setPslEntries(prev => [...prev, ...data]);
          setFilteredPslEntries(prev => [...prev, ...data]);
        } else {
          // Replace entries for initial load or refresh
          setPslEntries(data);
          setFilteredPslEntries(data);
        }
        
        // Check if there are more entries to load
        setPslHasMore(data.length === 20);
      } else {
        showNotification('error', 'Failed to fetch PSL alphabet entries');
      }
    } catch (error) {
      console.error('Error fetching PSL entries:', error);
      showNotification('error', 'Error fetching PSL alphabet entries');
    } finally {
      if (page === 1) {
        setPslLoading(false);
      } else {
        setPslLoadingMore(false);
      }
    }
  };

  const loadMorePslEntries = async () => {
    if (pslLoadingMore || !pslHasMore) return;
    
    const nextPage = pslPage + 1;
    setPslPage(nextPage);
    await fetchPslEntries(nextPage, true);
  };

  const handleCreatePsl = async () => {
    try {
      const response = await fetch('/api/v1/psl-alphabet/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(pslFormData)
      });

      if (response.ok) {
        showNotification('success', 'PSL alphabet entry created successfully');
        setShowPslModal(false);
        resetPslForm();
        fetchPslEntries();
      } else {
        const error = await response.json();
        showNotification('error', error.detail || 'Failed to create entry');
      }
    } catch (error) {
      console.error('Error creating PSL entry:', error);
      showNotification('error', 'Error creating PSL alphabet entry');
    }
  };

  const handleUpdatePsl = async () => {
    if (!editingPslEntry) return;

    try {
      const response = await fetch(`/api/v1/psl-alphabet/${editingPslEntry.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(pslFormData)
      });

      if (response.ok) {
        showNotification('success', 'PSL alphabet entry updated successfully');
        setShowPslModal(false);
        setEditingPslEntry(null);
        resetPslForm();
        fetchPslEntries();
      } else {
        const error = await response.json();
        showNotification('error', error.detail || 'Failed to update entry');
      }
    } catch (error) {
      console.error('Error updating PSL entry:', error);
      showNotification('error', 'Error updating PSL alphabet entry');
    }
  };

  const handleDeletePsl = async (id: number, letter: string) => {
    if (!confirm(`Are you sure you want to delete the PSL entry for letter "${letter}"?`)) {
      return;
    }

    try {
      const response = await fetch(`/api/v1/psl-alphabet/${id}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        showNotification('success', 'PSL alphabet entry deleted successfully');
        fetchPslEntries();
      } else {
        showNotification('error', 'Failed to delete entry');
      }
    } catch (error) {
      console.error('Error deleting PSL entry:', error);
      showNotification('error', 'Error deleting PSL alphabet entry');
    }
  };

  const togglePslStatus = async (id: number) => {
    try {
      const response = await fetch(`/api/v1/psl-alphabet/${id}/toggle-status`, {
        method: 'PATCH',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        showNotification('success', 'Entry status updated successfully');
        fetchPslEntries();
      } else {
        showNotification('error', 'Failed to update entry status');
      }
    } catch (error) {
      console.error('Error updating PSL entry status:', error);
      showNotification('error', 'Error updating entry status');
    }
  };

  const handleEditPsl = (entry: PSLAlphabetEntry) => {
    setEditingPslEntry(entry);
    setPslFormData({
      letter: entry.letter,
      file_path: entry.file_path,
      label: entry.label,
      difficulty: entry.difficulty,
      description: entry.description || '',
      is_active: entry.is_active
    });
    setShowPslModal(true);
  };

  const resetPslForm = () => {
    setPslFormData({
      letter: '',
      file_path: '',
      label: '',
      difficulty: 'easy',
      description: '',
      is_active: true
    });
  };

  // Filter PSL entries based on search and filters
  React.useEffect(() => {
    // When filters change, reset pagination and fetch fresh data
    if (activeTab === 'psl') {
      setPslPage(1);
      setPslHasMore(true);
      fetchPslEntries(1, false);
    }
  }, [pslSearchTerm, pslDifficultyFilter, pslStatusFilter]);

  // Load PSL entries when tab is active
  React.useEffect(() => {
    if (activeTab === 'psl' && pslEntries.length === 0) {
      fetchPslEntries();
    }
  }, [activeTab, token]);

  // Load contact messages when tab is active
  React.useEffect(() => {
    if (activeTab === 'contact' && contactMessages.length === 0) {
      fetchContactMessages();
    }
  }, [activeTab, token]);

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(userSearchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(userSearchTerm.toLowerCase()) ||
                         (user.full_name && user.full_name.toLowerCase().includes(userSearchTerm.toLowerCase()));
    
    const matchesRole = userRoleFilter === 'all' || user.role === userRoleFilter;
    const matchesStatus = userStatusFilter === 'all' || 
                         (userStatusFilter === 'active' && user.is_active) ||
                         (userStatusFilter === 'inactive' && !user.is_active);
    
    return matchesSearch && matchesRole && matchesStatus;
  });

  // Pagination
  const totalPages = Math.ceil(filteredUsers.length / usersPerPage);
  const paginatedUsers = filteredUsers.slice(
    (currentPage - 1) * usersPerPage,
    currentPage * usersPerPage
  );

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="relative max-w-md mx-auto text-center">
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-200">
            <div className="bg-red-500 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4 shadow-lg">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-red-600 mb-4">
              🚫 Access Denied
            </h2>
            <p className="text-gray-600 mb-6">You need administrator privileges to access this portal.</p>
            <button
              onClick={() => navigate('/dashboard')}
              className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-3 rounded-lg transition-all duration-300 shadow-md"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-gray-200 mx-auto"></div>
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-500 border-t-transparent absolute top-0 left-1/2 transform -translate-x-1/2"></div>
          </div>
          <div className="mt-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              Loading Admin Portal
            </h2>
            <p className="text-gray-600">Initializing system dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      
      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm ${
          notification.type === 'success' 
            ? 'bg-green-500 text-white' 
            : 'bg-red-500 text-white'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              {notification.type === 'success' ? (
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
              <span className="text-sm font-medium">{notification.message}</span>
            </div>
            <button
              onClick={() => setNotification(null)}
              className="ml-3 text-white hover:text-gray-200"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center shadow-md">
                  <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-800">
                    Admin Dashboard
                  </h1>
                  <p className="text-gray-600 text-sm font-medium">ASLense Management Portal</p>
                </div>
              </div>
              <span className="ml-6 px-4 py-2 bg-red-500 text-white text-sm font-bold rounded-full shadow-md">
                ⚡ Administrator
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <Link
                to="/csv"
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300 flex items-center space-x-2 shadow-md"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                </svg>
                <span className="font-medium">CSV Manager</span>
              </Link>
              <button
                onClick={() => navigate('/dashboard')}
                className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-all duration-300"
              >
                <span className="font-medium">← Back to Dashboard</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Welcome Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">
            System Administration Center
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Monitor, manage, and optimize your ASLense platform
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-lg mb-12 border border-gray-200">
          <nav className="flex space-x-1 p-3">
            {[
              { id: 'overview', label: 'Overview', icon: '📊', color: 'blue' },
              { id: 'users', label: 'User Management', icon: '👥', color: 'green' },
              { id: 'videos', label: 'Video Management', icon: '🎥', color: 'purple' },
              { id: 'psl', label: 'PSL Alphabet', icon: '🤟', color: 'indigo' },
              { id: 'contact', label: 'Contact Messages', icon: '✉️', color: 'pink' },
              { id: 'metrics', label: 'System Metrics', icon: '📈', color: 'orange' },
              { id: 'tools', label: 'Admin Tools', icon: '🛠️', color: 'gray' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex-1 flex items-center justify-center space-x-3 py-4 px-6 rounded-lg font-bold text-sm transition-all duration-300 ${
                  activeTab === tab.id
                    ? `bg-${tab.color}-500 text-white shadow-md`
                    : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
                }`}
              >
                <span className="text-xl">{tab.icon}</span>
                <span className="hidden sm:block">{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-12">
            {/* Stats Grid */}
            {stats && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-center">
                    <div className="bg-blue-500 rounded-lg p-3 shadow-md">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-800">{stats.totalUsers}</p>
                      <p className="text-gray-600 font-medium">Total Users</p>
                      <div className="mt-1 text-sm text-green-600 font-medium">
                        +{stats.activeUsers || 0} active
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-center">
                    <div className="bg-green-500 rounded-lg p-3 shadow-md">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-800">{stats.totalPredictions.toLocaleString()}</p>
                      <p className="text-gray-600 font-medium">Predictions</p>
                      <div className="mt-1 text-sm text-green-600 font-medium">
                        {stats.averageAccuracy.toFixed(1)}% avg accuracy
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-center">
                    <div className="bg-orange-500 rounded-lg p-3 shadow-md">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-800">{stats.averageAccuracy.toFixed(1)}%</p>
                      <p className="text-gray-600 font-medium">Accuracy Rate</p>
                      <div className="mt-1 text-sm text-green-600 font-medium">
                        System performance
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                  <div className="flex items-center">
                    <div className="bg-purple-500 rounded-lg p-3 shadow-md">
                      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-3xl font-bold text-gray-800">{stats.totalVideos}</p>
                      <p className="text-gray-600 font-medium">Training Videos</p>
                      <div className="mt-1 text-sm text-green-600 font-medium">
                        Learning content
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* System Status Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                  <div className="bg-blue-500 rounded-lg p-2 mr-3">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  System Performance
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600 font-medium">Average Accuracy</span>
                    <span className="text-xl font-bold text-gray-800">{stats?.averageAccuracy?.toFixed(1) || '0.0'}%</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600 font-medium">Active Users</span>
                    <span className="text-xl font-bold text-green-600">{stats?.activeUsers || 0}</span>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                  <div className="bg-green-500 rounded-lg p-2 mr-3">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  Quick Actions
                </h3>
                <div className="grid grid-cols-1 gap-4">
                  <Link
                    to="/csv"
                    className="flex items-center p-4 bg-blue-50 rounded-lg border border-blue-200 hover:bg-blue-100 transition-all duration-300"
                  >
                    <div className="bg-blue-500 rounded-lg p-2">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <h4 className="font-bold text-gray-800">CSV Manager</h4>
                      <p className="text-sm text-gray-600">Import/Export video data</p>
                    </div>
                  </Link>

                  <button
                    onClick={() => setActiveTab('users')}
                    className="flex items-center p-4 bg-green-50 rounded-lg border border-green-200 hover:bg-green-100 transition-all duration-300"
                  >
                    <div className="bg-green-500 rounded-lg p-2">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <h4 className="font-bold text-gray-800">Manage Users</h4>
                      <p className="text-sm text-gray-600">View and edit user accounts</p>
                    </div>
                  </button>
                </div>
              </div>
            </div>

            {/* User Management Summary */}
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                <div className="bg-green-500 rounded-lg p-2 mr-3">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
                User Demographics
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="text-3xl font-bold text-blue-600">
                    {users.filter(u => u.role === 'user').length}
                  </div>
                  <div className="text-sm text-gray-600 font-medium">Regular Users</div>
                </div>
                <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200">
                  <div className="text-3xl font-bold text-red-600">
                    {users.filter(u => u.role === 'admin').length}
                  </div>
                  <div className="text-sm text-gray-600 font-medium">Admin Users</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <div className="text-3xl font-bold text-yellow-600">
                    {users.filter(u => !u.is_active).length}
                  </div>
                  <div className="text-sm text-gray-600 font-medium">Inactive Users</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* User Management Tab */}
        {activeTab === 'users' && (
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-xl font-bold text-gray-800 flex items-center">
                <div className="bg-green-500 rounded-lg p-2 mr-3">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                  </svg>
                </div>
                User Management
              </h3>
              <button 
                onClick={() => setShowCreateUserModal(true)}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all duration-300 flex items-center space-x-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Add New User</span>
              </button>
            </div>

            {/* Search and Filter Controls */}
            <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="flex-1">
                  <input
                    type="text"
                    placeholder="Search users by name, username, or email..."
                    value={userSearchTerm}
                    onChange={(e) => setUserSearchTerm(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div className="flex gap-3">
                  <select
                    value={userRoleFilter}
                    onChange={(e) => setUserRoleFilter(e.target.value as 'all' | 'admin' | 'user')}
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Roles</option>
                    <option value="admin">Admin</option>
                    <option value="user">User</option>
                  </select>
                  <select
                    value={userStatusFilter}
                    onChange={(e) => setUserStatusFilter(e.target.value as 'all' | 'active' | 'inactive')}
                    className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Status</option>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                  </select>
                </div>
              </div>
            </div>
            
            <div className="p-6">
              {filteredUsers.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                    </svg>
                  </div>
                  <h4 className="text-lg font-medium text-gray-900 mb-2">
                    {users.length === 0 ? 'No users found' : 'No users match your filters'}
                  </h4>
                  <p className="text-gray-500">
                    {users.length === 0 ? 'Users will appear here once they register' : 'Try adjusting your search or filter criteria'}
                  </p>
                </div>
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Role</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Login</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {paginatedUsers.map((user) => (
                          <tr key={user.id} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                                  <span className="text-white font-bold text-sm">
                                    {user.username.charAt(0).toUpperCase()}
                                  </span>
                                </div>                              <div className="ml-4">
                                <div className="text-sm font-medium text-gray-900">{user.username}</div>
                                <div className="text-sm text-gray-500">{user.email}</div>
                                {user.full_name && <div className="text-sm text-gray-500">{user.full_name}</div>}
                              </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                                user.role === 'admin' 
                                  ? 'bg-red-100 text-red-800' 
                                  : 'bg-blue-100 text-blue-800'
                              }`}>
                                {user.role.toUpperCase()}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">                            <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                              user.is_active 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {user.is_active ? 'ACTIVE' : 'INACTIVE'}
                            </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">                            {new Date(user.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                              <div className="flex space-x-2">                              <button
                                onClick={() => toggleUserStatus(user.id, user.is_active)}
                                className={`px-3 py-1 rounded text-xs font-medium transition-all duration-300 ${
                                  user.is_active
                                    ? 'bg-red-100 text-red-700 hover:bg-red-200'
                                    : 'bg-green-100 text-green-700 hover:bg-green-200'
                                }`}
                              >
                                {user.is_active ? 'Deactivate' : 'Activate'}
                              </button>
                                <button 
                                  onClick={() => openEditModal(user)}
                                  className="px-3 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium hover:bg-blue-200 transition-all duration-300"
                                >
                                  Edit
                                </button>
                                <button 
                                  onClick={() => deleteUser(user.id)}
                                  className="px-3 py-1 bg-red-100 text-red-700 rounded text-xs font-medium hover:bg-red-200 transition-all duration-300"
                                >
                                  Delete
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pagination */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between mt-6 px-4 py-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-gray-600">
                        Showing {((currentPage - 1) * usersPerPage) + 1} to {Math.min(currentPage * usersPerPage, filteredUsers.length)} of {filteredUsers.length} users
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                          disabled={currentPage === 1}
                          className="px-3 py-1 bg-white border border-gray-300 rounded text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Previous
                        </button>
                        {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                          <button
                            key={page}
                            onClick={() => setCurrentPage(page)}
                            className={`px-3 py-1 text-sm font-medium rounded transition-all duration-300 ${
                              currentPage === page
                                ? 'bg-blue-500 text-white'
                                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                            }`}
                          >
                            {page}
                          </button>
                        ))}
                        <button
                          onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                          disabled={currentPage === totalPages}
                          className="px-3 py-1 bg-white border border-gray-300 rounded text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Videos Tab */}
        {activeTab === 'videos' && (
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-xl font-bold text-gray-800 flex items-center">
                <div className="bg-purple-500 rounded-lg p-2 mr-3">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                Video Management
              </h3>
              <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-all duration-300 flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Add New Video</span>
              </button>
            </div>
            
            <div className="p-6">
              {videos.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h4 className="text-lg font-medium text-gray-900 mb-2">No videos found</h4>
                  <p className="text-gray-500">Upload your first ASL video to get started</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-6">
                  {videos.map((video) => (
                    <div key={video.id} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all duration-300">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="w-16 h-16 bg-purple-500 rounded-lg flex items-center justify-center">
                            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <div>
                            <h4 className="text-lg font-bold text-gray-800">{video.title}</h4>
                            <p className="text-gray-600 text-sm">{video.category} • {video.difficulty}</p>
                            <p className="text-gray-500 text-xs">Word: {video.word}</p>
                            {video.duration && (
                              <p className="text-gray-500 text-xs">Duration: {video.duration}s</p>
                            )}
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <button className="px-3 py-1 bg-blue-100 text-blue-700 rounded text-sm font-medium hover:bg-blue-200 transition-all duration-300">
                            Edit
                          </button>
                          <button className="px-3 py-1 bg-red-100 text-red-700 rounded text-sm font-medium hover:bg-red-200 transition-all duration-300">
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* PSL Alphabet Management Tab */}
        {activeTab === 'psl' && (
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-xl font-bold text-gray-800 flex items-center">
                <div className="bg-indigo-500 rounded-lg p-2 mr-3">
                  <span className="text-white text-lg">🤟</span>
                </div>
                PSL Alphabet Management
              </h3>
              <button 
                onClick={() => {
                  setPslFormData({
                    letter: '',
                    file_path: '',
                    label: '',
                    difficulty: 'easy',
                    description: '',
                    is_active: true
                  });
                  setEditingPslEntry(null);
                  setShowPslModal(true);
                }}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-all duration-300 flex items-center space-x-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span>Add New Entry</span>
              </button>
            </div>

            {/* Search and Filters */}
            <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
              <div className="flex flex-col lg:flex-row gap-4 items-center">
                <div className="relative flex-1">
                  <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input
                    type="text"
                    placeholder="Search letters, labels, or files..."
                    value={pslSearchTerm}
                    onChange={(e) => setPslSearchTerm(e.target.value)}
                    className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent w-full"
                  />
                </div>

                <select
                  value={pslDifficultyFilter}
                  onChange={(e) => setPslDifficultyFilter(e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                >
                  <option value="">All Difficulties</option>
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>

                <select
                  value={pslStatusFilter}
                  onChange={(e) => setPslStatusFilter(e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                </select>
              </div>
            </div>
            
            <div className="p-6">
              {pslLoading ? (
                <div className="text-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto"></div>
                  <p className="mt-2 text-gray-600">Loading PSL entries...</p>
                </div>
              ) : filteredPslEntries.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-16 h-16 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">🤟</span>
                  </div>
                  <h4 className="text-lg font-medium text-gray-900 mb-2">No PSL entries found</h4>
                  <p className="text-gray-500">Add your first PSL alphabet entry to get started</p>
                </div>
              ) : (
                <div 
                  className="grid grid-cols-1 gap-4 max-h-96 overflow-y-auto pr-2"
                  onScroll={(e) => {
                    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
                    // Load more when scrolled to bottom
                    if (scrollHeight - scrollTop === clientHeight && pslHasMore && !pslLoadingMore) {
                      loadMorePslEntries();
                    }
                  }}
                >
                  {filteredPslEntries.map((entry) => (
                    <div key={entry.id} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all duration-300">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                          <div className="w-16 h-16 bg-indigo-500 rounded-lg flex items-center justify-center">
                            <span className="text-white text-xl font-bold">{entry.letter.charAt(0)}</span>
                          </div>
                          <div>
                            <h4 className="text-lg font-bold text-gray-800">{entry.letter}</h4>
                            <p className="text-gray-600 text-sm">
                              {entry.label} • 
                              <span className={`ml-1 px-2 py-1 rounded-full text-xs font-semibold ${
                                entry.difficulty === 'easy' ? 'bg-green-100 text-green-800' :
                                entry.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {entry.difficulty}
                              </span>
                            </p>
                            <p className="text-gray-500 text-xs">File: {entry.file_path}</p>
                            {entry.description && (
                              <p className="text-gray-500 text-xs mt-1">{entry.description}</p>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => togglePslStatus(entry.id)}
                            className={`px-3 py-1 rounded text-xs font-medium transition-all duration-300 ${
                              entry.is_active
                                ? 'bg-green-100 text-green-700 hover:bg-green-200'
                                : 'bg-red-100 text-red-700 hover:bg-red-200'
                            }`}
                          >
                            {entry.is_active ? 'Active' : 'Inactive'}
                          </button>
                          <button 
                            onClick={() => handleEditPsl(entry)}
                            className="px-3 py-1 bg-blue-100 text-blue-700 rounded text-sm font-medium hover:bg-blue-200 transition-all duration-300"
                          >
                            Edit
                          </button>
                          <button 
                            onClick={() => handleDeletePsl(entry.id, entry.letter)}
                            className="px-3 py-1 bg-red-100 text-red-700 rounded text-sm font-medium hover:bg-red-200 transition-all duration-300"
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {/* Loading More Indicator */}
                  {pslLoadingMore && (
                    <div className="text-center py-4">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600 mx-auto"></div>
                      <p className="mt-2 text-sm text-gray-600">Loading more entries...</p>
                    </div>
                  )}
                  
                  {/* End of Results Indicator */}
                  {!pslHasMore && filteredPslEntries.length > 0 && (
                    <div className="text-center py-4">
                      <p className="text-sm text-gray-500">No more entries to load</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Contact Messages Tab */}
        {activeTab === 'contact' && (
          <div className="space-y-8">
            {/* Contact Stats Cards */}
            {contactStats && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                  <div className="flex items-center">
                    <div className="bg-pink-500 rounded-lg p-3">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-gray-800">{contactStats.total}</p>
                      <p className="text-gray-600">Total Messages</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                  <div className="flex items-center">
                    <div className="bg-red-500 rounded-lg p-3">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-gray-800">{contactStats.unread}</p>
                      <p className="text-gray-600">Unread</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                  <div className="flex items-center">
                    <div className="bg-yellow-500 rounded-lg p-3">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-gray-800">{contactStats.read}</p>
                      <p className="text-gray-600">Read</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                  <div className="flex items-center">
                    <div className="bg-green-500 rounded-lg p-3">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div className="ml-4">
                      <p className="text-2xl font-bold text-gray-800">{contactStats.replied}</p>
                      <p className="text-gray-600">Replied</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Contact Messages Management */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900 flex items-center">
                  <div className="w-4 h-4 bg-pink-500 rounded-full mr-3"></div>
                  Contact Messages
                </h2>
                
                {/* Filter Controls */}
                <div className="flex space-x-3">
                  <select
                    value={messageStatusFilter}
                    onChange={(e) => {
                      setMessageStatusFilter(e.target.value);
                      fetchContactMessages(e.target.value);
                    }}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-transparent"
                  >
                    <option value="all">All Messages</option>
                    <option value="unread">Unread</option>
                    <option value="read">Read</option>
                    <option value="replied">Replied</option>
                  </select>
                </div>
              </div>

              {/* Messages List */}
              {contactLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pink-600 mx-auto"></div>
                  <p className="mt-2 text-gray-600">Loading contact messages...</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Sender
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Subject
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {contactMessages.map((message) => (
                        <tr key={message.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex flex-col">
                              <div className="text-sm font-medium text-gray-900">{message.name}</div>
                              <div className="text-sm text-gray-500">{message.email}</div>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div className="text-sm text-gray-900 max-w-xs truncate">{message.subject}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              message.status === 'unread' 
                                ? 'bg-red-100 text-red-800'
                                : message.status === 'read'
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {message.status}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {new Date(message.created_at).toLocaleDateString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <div className="flex space-x-2">
                              <button
                                onClick={() => openMessageModal(message)}
                                className="text-pink-600 hover:text-pink-900"
                                title="View message"
                              >
                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                </svg>
                              </button>
                              {message.status === 'unread' && (
                                <button
                                  onClick={() => markMessageAsRead(message.id)}
                                  className="text-yellow-600 hover:text-yellow-900"
                                  title="Mark as read"
                                >
                                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                  </svg>
                                </button>
                              )}
                              <button
                                onClick={() => deleteContactMessage(message.id)}
                                className="text-red-600 hover:text-red-900"
                                title="Delete message"
                              >
                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  
                  {contactMessages.length === 0 && (
                    <div className="text-center py-8">
                      <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                      </svg>
                      <p className="mt-2 text-gray-500">No contact messages found</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* System Metrics Tab */}
        {activeTab === 'metrics' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                  <div className="bg-orange-500 rounded-lg p-2 mr-3">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  Performance Metrics
                </h3>
                <div className="space-y-4">
                  {metrics && (
                    <>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-gray-600 font-medium">System Accuracy</span>
                        <span className="text-xl font-bold text-green-600">
                          {metrics.predictionAccuracy.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-gray-600 font-medium">Avg Session Time</span>
                        <span className="text-xl font-bold text-gray-800">
                          {Math.round(metrics.averageSessionTime)} min
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-gray-600 font-medium">Daily Active Users</span>
                        <span className="text-xl font-bold text-blue-600">
                          {metrics.dailyActiveUsers}
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-gray-600 font-medium">Total Sessions</span>
                        <span className="text-xl font-bold text-gray-800">
                          {metrics.totalSessions.toLocaleString()}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                  <div className="bg-green-500 rounded-lg p-2 mr-3">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                  </div>
                  System Health
                </h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600 font-medium">Database Status</span>
                    <span className="px-3 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                      ✓ Healthy
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600 font-medium">API Status</span>
                    <span className="px-3 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                      ✓ Online
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600 font-medium">Model Status</span>
                    <span className="px-3 py-1 bg-green-100 text-green-800 text-sm font-medium rounded-full">
                      ✓ Active
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600 font-medium">Storage Usage</span>
                    <div className="text-right">
                      <span className="text-gray-800 font-medium">45.2 GB / 100 GB</span>
                      <div className="w-32 h-2 bg-gray-200 rounded-full mt-1 overflow-hidden">
                        <div className="w-11/24 h-full bg-green-500 rounded-full"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tools Tab */}
        {activeTab === 'tools' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                <div className="bg-gray-500 rounded-lg p-2 mr-3">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                Admin Tools
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Link
                  to="/csv"
                  className="group block p-6 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-lg transition-all duration-200"
                >
                  <div className="flex items-center space-x-4">
                    <div className="bg-blue-500 rounded-lg p-3 group-hover:scale-105 transition-transform duration-200">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-lg font-bold text-gray-800 mb-1">CSV Data Manager</h4>
                      <p className="text-gray-600 text-sm">Import and export video datasets for training and analysis</p>
                    </div>
                  </div>
                </Link>

                <div className="group p-6 bg-purple-50 hover:bg-purple-100 border border-purple-200 rounded-lg transition-all duration-200">
                  <div className="flex items-center space-x-4">
                    <div className="bg-purple-500 rounded-lg p-3 group-hover:scale-105 transition-transform duration-200">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-lg font-bold text-gray-800 mb-1">System Settings</h4>
                      <p className="text-gray-600 text-sm">Configure application settings and system parameters</p>
                    </div>
                  </div>
                </div>

                <div className="group p-6 bg-orange-50 hover:bg-orange-100 border border-orange-200 rounded-lg transition-all duration-200">
                  <div className="flex items-center space-x-4">
                    <div className="bg-orange-500 rounded-lg p-3 group-hover:scale-105 transition-transform duration-200">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-lg font-bold text-gray-800 mb-1">Analytics Dashboard</h4>
                      <p className="text-gray-600 text-sm">View detailed analytics and performance reports</p>
                    </div>
                  </div>
                </div>

                <div className="group p-6 bg-green-50 hover:bg-green-100 border border-green-200 rounded-lg transition-all duration-200">
                  <div className="flex items-center space-x-4">
                    <div className="bg-green-500 rounded-lg p-3 group-hover:scale-105 transition-transform duration-200">
                      <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-lg font-bold text-gray-800 mb-1">System Maintenance</h4>
                      <p className="text-gray-600 text-sm">Database maintenance and system optimization tools</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Create User Modal */}
      {showCreateUserModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-xl font-bold text-gray-800 mb-4">Create New User</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={createUserData.username}
                  onChange={(e) => setCreateUserData({ ...createUserData, username: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter username"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={createUserData.email}
                  onChange={(e) => setCreateUserData({ ...createUserData, email: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter email"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input
                  type="password"
                  value={createUserData.password}
                  onChange={(e) => setCreateUserData({ ...createUserData, password: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter password"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Full Name (Optional)</label>
                <input
                  type="text"
                  value={createUserData.full_name}
                  onChange={(e) => setCreateUserData({ ...createUserData, full_name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter full name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select
                  value={createUserData.role}
                  onChange={(e) => setCreateUserData({ ...createUserData, role: e.target.value as 'admin' | 'user' })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowCreateUserModal(false)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-all duration-300"
              >
                Cancel
              </button>
              <button
                onClick={createUser}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all duration-300"
              >
                Create User
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {showEditUserModal && selectedUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-xl font-bold text-gray-800 mb-4">Edit User</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={editUserData.username || ''}
                  onChange={(e) => setEditUserData({ ...editUserData, username: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={editUserData.email || ''}
                  onChange={(e) => setEditUserData({ ...editUserData, email: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
                <input
                  type="text"
                  value={editUserData.full_name || ''}
                  onChange={(e) => setEditUserData({ ...editUserData, full_name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select
                  value={editUserData.role || 'user'}
                  onChange={(e) => setEditUserData({ ...editUserData, role: e.target.value as 'admin' | 'user' })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="user">User</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={editUserData.is_active ?? true}
                  onChange={(e) => setEditUserData({ ...editUserData, is_active: e.target.checked })}
                  className="mr-2"
                />
                <label className="text-sm font-medium text-gray-700">Active User</label>
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowEditUserModal(false)}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-all duration-300"
              >
                Cancel
              </button>
              <button
                onClick={updateUser}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-300"
              >
                Update User
              </button>
            </div>
          </div>
        </div>
      )}

      {/* PSL Add/Edit Modal */}
      {showPslModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">
                {editingPslEntry ? 'Edit PSL Entry' : 'Add New PSL Entry'}
              </h3>
              <button
                onClick={() => {
                  setShowPslModal(false);
                  setEditingPslEntry(null);
                  resetPslForm();
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Letter
                </label>
                <input
                  type="text"
                  value={pslFormData.letter}
                  onChange={(e) => setPslFormData({...pslFormData, letter: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
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
                  value={pslFormData.label}
                  onChange={(e) => setPslFormData({...pslFormData, label: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Alif"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  File Path
                </label>
                <input
                  type="text"
                  value={pslFormData.file_path}
                  onChange={(e) => setPslFormData({...pslFormData, file_path: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  placeholder="Alif-Original_s0251-02alif.jpg"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Difficulty
                </label>
                <select
                  value={pslFormData.difficulty}
                  onChange={(e) => setPslFormData({...pslFormData, difficulty: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
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
                  value={pslFormData.description}
                  onChange={(e) => setPslFormData({...pslFormData, description: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  rows={3}
                  placeholder="How to sign this letter..."
                />
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="psl_is_active"
                  checked={pslFormData.is_active}
                  onChange={(e) => setPslFormData({...pslFormData, is_active: e.target.checked})}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
                <label htmlFor="psl_is_active" className="ml-2 block text-sm text-gray-700">
                  Active
                </label>
              </div>
            </div>

            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => {
                  setShowPslModal(false);
                  setEditingPslEntry(null);
                  resetPslForm();
                }}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={editingPslEntry ? handleUpdatePsl : handleCreatePsl}
                className="flex items-center space-x-2 bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
              >
                <span>{editingPslEntry ? 'Update' : 'Create'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Contact Message Modal */}
      {showMessageModal && selectedMessage && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-3xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center p-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Contact Message Details</h3>
              <button
                onClick={() => {
                  setShowMessageModal(false);
                  setSelectedMessage(null);
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Message Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                  <p className="text-gray-900">{selectedMessage.name}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                  <p className="text-gray-900">{selectedMessage.email}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
                  <p className="text-gray-900">{selectedMessage.subject}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    selectedMessage.status === 'unread' 
                      ? 'bg-red-100 text-red-800'
                      : selectedMessage.status === 'read'
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {selectedMessage.status}
                  </span>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Date Received</label>
                  <p className="text-gray-900">{new Date(selectedMessage.created_at).toLocaleString()}</p>
                </div>
              </div>

              {/* Message Content */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Message</label>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-900 whitespace-pre-wrap">{selectedMessage.message}</p>
                </div>
              </div>

              {/* Admin Notes */}
              {selectedMessage.admin_notes && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Admin Notes</label>
                  <div className="bg-blue-50 rounded-lg p-4">
                    <p className="text-gray-900 whitespace-pre-wrap">{selectedMessage.admin_notes}</p>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
                <button
                  onClick={() => {
                    setShowMessageModal(false);
                    setSelectedMessage(null);
                  }}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                >
                  Close
                </button>
                
                {selectedMessage.status !== 'replied' && (
                  <button
                    onClick={() => {
                      const adminNotes = prompt('Add admin notes (optional):') || '';
                      markMessageAsReplied(selectedMessage.id, adminNotes);
                      setShowMessageModal(false);
                      setSelectedMessage(null);
                    }}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Mark as Replied
                  </button>
                )}
                
                <button
                  onClick={() => {
                    deleteContactMessage(selectedMessage.id);
                    setShowMessageModal(false);
                    setSelectedMessage(null);
                  }}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Delete Message
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
