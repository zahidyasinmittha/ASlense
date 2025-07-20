// Session management for translation module
// Expires on browser close, tab close, or logout - NOT on refresh

export class TranslationSessionManager {
  private static instance: TranslationSessionManager;
  private sessionId: string | null = null;
  private sessionData: any = null;
  private beforeUnloadHandler: (() => void) | null = null;
  private visibilityChangeHandler: (() => void) | null = null;
  private heartbeatInterval: number | null = null;

  private constructor() {}

  static getInstance(): TranslationSessionManager {
    if (!TranslationSessionManager.instance) {
      TranslationSessionManager.instance = new TranslationSessionManager();
    }
    return TranslationSessionManager.instance;
  }

  startSession(data: any): string {
    // Generate unique session ID
    this.sessionId = `translate_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize session data
    this.sessionData = {
      ...data,
      startTime: Date.now(),
      translations: [],
      totalTranslations: 0,
      correctTranslations: 0,
      accuracySum: 0,
      sessionActive: true,
      lastActivity: Date.now()
    };

    // Store in sessionStorage (expires on tab close)
    sessionStorage.setItem(`translation_session_${this.sessionId}`, JSON.stringify(this.sessionData));
    
    // Set up session expiry handlers
    this.setupSessionExpiry();
    
    // Start heartbeat to track session activity
    this.startHeartbeat();
    
    return this.sessionId;
  }

  private setupSessionExpiry() {
    // Handle browser/tab close
    this.beforeUnloadHandler = () => {
      this.endSession('browser_close');
    };
    window.addEventListener('beforeunload', this.beforeUnloadHandler);

    // Handle tab visibility changes (background/foreground)
    this.visibilityChangeHandler = () => {
      if (document.hidden) {
        // Tab is hidden - start timeout to end session if not visible for too long
        setTimeout(() => {
          if (document.hidden && this.sessionId) {
            this.endSession('tab_hidden');
          }
        }, 30000); // 30 seconds timeout
      } else {
        // Tab is visible again - update last activity
        this.updateActivity();
      }
    };
    document.addEventListener('visibilitychange', this.visibilityChangeHandler);
  }

  private startHeartbeat() {
    // Update session every 5 seconds when active
    this.heartbeatInterval = window.setInterval(() => {
      if (this.sessionId && this.sessionData) {
        this.sessionData.lastActivity = Date.now();
        sessionStorage.setItem(`translation_session_${this.sessionId}`, JSON.stringify(this.sessionData));
      }
    }, 5000);
  }

  addTranslation(translation: any) {
    if (!this.sessionId || !this.sessionData) return;

    this.sessionData.translations.push({
      ...translation,
      timestamp: Date.now()
    });
    
    this.sessionData.totalTranslations++;
    this.sessionData.accuracySum += translation.confidence;
    
    if (translation.isCorrect) {
      this.sessionData.correctTranslations++;
    }
    
    this.sessionData.lastActivity = Date.now();
    
    // Update sessionStorage
    sessionStorage.setItem(`translation_session_${this.sessionId}`, JSON.stringify(this.sessionData));
  }

  getSessionStats() {
    if (!this.sessionData) return null;

    const sessionTime = Math.floor((Date.now() - this.sessionData.startTime) / 1000);
    const avgAccuracy = this.sessionData.totalTranslations > 0 
      ? (this.sessionData.accuracySum / this.sessionData.totalTranslations) * 100 
      : 0;
    const correctnessRate = this.sessionData.totalTranslations > 0
      ? (this.sessionData.correctTranslations / this.sessionData.totalTranslations) * 100
      : 0;

    return {
      translations: this.sessionData.totalTranslations,
      accuracy: Math.round(avgAccuracy * 100) / 100,
      accuracy_percentage: Math.round(correctnessRate * 100) / 100,
      session_time: sessionTime,
      correct_translations: this.sessionData.correctTranslations,
      model_type: this.sessionData.model_type || 'mediapipe',
      input_mode: this.sessionData.input_mode || 'word'
    };
  }

  getRecentTranslations(limit: number = 10) {
    if (!this.sessionData) return [];
    
    return this.sessionData.translations
      .slice(-limit)
      .reverse()
      .map((t: any) => ({
        predicted_text: t.predicted_text,
        target_text: t.target_text,
        confidence: t.confidence,
        is_correct: t.isCorrect,
        model_used: this.sessionData.model_type,
        input_mode: this.sessionData.input_mode,
        timestamp: new Date(t.timestamp).toISOString()
      }));
  }

  updateActivity() {
    if (this.sessionData) {
      this.sessionData.lastActivity = Date.now();
      sessionStorage.setItem(`translation_session_${this.sessionId}`, JSON.stringify(this.sessionData));
    }
  }

  endSession(reason: string = 'manual'): any {
    if (!this.sessionId || !this.sessionData) return null;

    const finalStats = this.getSessionStats();
    
    // Clean up event listeners
    if (this.beforeUnloadHandler) {
      window.removeEventListener('beforeunload', this.beforeUnloadHandler);
      this.beforeUnloadHandler = null;
    }
    
    if (this.visibilityChangeHandler) {
      document.removeEventListener('visibilitychange', this.visibilityChangeHandler);
      this.visibilityChangeHandler = null;
    }
    
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Store final session data in localStorage for history (persists across sessions)
    const historicalSessions = JSON.parse(localStorage.getItem('translation_history_sessions') || '[]');
    historicalSessions.unshift({
      sessionId: this.sessionId,
      endReason: reason,
      endTime: Date.now(),
      ...finalStats,
      translations: this.sessionData.translations
    });
    
    // Keep only last 10 sessions
    localStorage.setItem('translation_history_sessions', JSON.stringify(historicalSessions.slice(0, 10)));
    
    // Clear session storage
    sessionStorage.removeItem(`translation_session_${this.sessionId}`);
    
    // Reset instance
    this.sessionId = null;
    this.sessionData = null;
    
    return finalStats;
  }

  isSessionActive(): boolean {
    return this.sessionId !== null && this.sessionData !== null;
  }

  getCurrentSessionId(): string | null {
    return this.sessionId;
  }

  // Get historical sessions from localStorage
  getHistoricalSessions(): any[] {
    return JSON.parse(localStorage.getItem('translation_history_sessions') || '[]');
  }

  // Simulate translation result for demo
  simulateTranslation(modelType: string, inputMode: string, targetText?: string): any {
    const vocabulary = {
      word: ['hello', 'thank you', 'please', 'sorry', 'yes', 'no', 'good', 'bad', 'happy', 'sad', 'love', 'family'],
      sentence: [
        'hello how are you',
        'thank you very much', 
        'please wait a moment',
        'nice to meet you',
        'good morning everyone',
        'see you later today'
      ]
    };

    const words = vocabulary[inputMode as keyof typeof vocabulary] || vocabulary.word;
    const predictedText = words[Math.floor(Math.random() * words.length)];
    
    // Simulate model accuracy
    const baseAccuracy = {
      mediapipe: 0.92,
      openpose: 0.87,
      custom: 0.89
    }[modelType] || 0.85;
    
    const confidence = Math.max(0.6, Math.min(0.98, baseAccuracy + (Math.random() - 0.5) * 0.2));
    
    // Check if prediction matches target
    const isCorrect = targetText ? 
      predictedText.toLowerCase().trim() === targetText.toLowerCase().trim() :
      Math.random() > 0.3; // 70% chance of being "correct" when no target

    return {
      predicted_text: predictedText,
      confidence: confidence,
      isCorrect: isCorrect,
      target_text: targetText,
      model_used: modelType
    };
  }
}

// Auto-cleanup on logout
export const setupLogoutCleanup = (logoutCallback: () => void) => {
  const originalLogout = logoutCallback;
  return () => {
    // End translation session on logout
    const sessionManager = TranslationSessionManager.getInstance();
    if (sessionManager.isSessionActive()) {
      sessionManager.endSession('logout');
    }
    originalLogout();
  };
};

// Export singleton instance
export const translationSessionManager = TranslationSessionManager.getInstance();
