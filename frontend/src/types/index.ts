// Shared TypeScript interfaces and types

export interface Video {
  id: number;
  word: string;
  title: string;
  description: string;
  difficulty: string;
  duration: string;
  video_file: string;
  thumbnail: string;
  category: string;
}

export interface VideoSearchParams {
  query?: string;
  category?: string;
  difficulty?: string;
  page?: number;
  limit?: number;
}

export interface VideoSearchResult {
  videos: Video[];
  total: number;
  page: number;
}

export interface PathSettings {
  custom_video_dir: string | null;
  custom_thumbnail_dir: string | null;
  default_video_dir: string;
  default_thumbnail_dir: string;
}

export interface ImportResult {
  success: boolean;
  imported?: number;
  failed?: number;
  failed_videos?: string[];
  message?: string;
  error?: string;
}

export interface DeleteResult {
  success: boolean;
  message?: string;
  error?: string;
}

export interface ApiError {
  detail?: string;
  error?: string;
  message?: string;
}

export interface MessageState {
  type: 'success' | 'error' | 'info' | 'warning';
  text: string;
  details?: string[];
}

export interface CSVImportProgress {
  total: number;
  processed: number;
  successful: number;
  failed: number;
  current?: string;
}

// API Response types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  error?: string;
}

// Configuration types
export interface AppConfig {
  backendUrl: string;
  supportedVideoFormats: string[];
  maxFileSize: number;
  thumbnailPath: string;
}

// Component prop types
export interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: string;
}

export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  onClick?: () => void;
  children: React.ReactNode;
}

export interface InputProps {
  label?: string;
  placeholder?: string;
  value: string;
  onChange: (value: string) => void;
  error?: string;
  disabled?: boolean;
  required?: boolean;
}
