// Theme mode type
export type ThemeMode = 'light' | 'dark' | 'system';

// Application configuration type
export interface AppConfig {
  companyName: string;
  pageTitle: string;
  pageDescription: string;
  supportsChatInput: boolean;
  supportsVideoInput: boolean;
  supportsScreenShare: boolean;
  isPreConnectBufferEnabled: boolean;
  logo: string;
  accent: string;
  logoDark: string;
  accentDark: string;
  startButtonText: string;
  agentName: string;
  sandboxId?: string;
}

// Chat message type
export interface ChatMessage {
  id: string;
  message: string;
  timestamp: number;
  name: string;
  isSelf: boolean;
  isFinal?: boolean;
}

// Transcription data type
export interface TranscriptionData {
  text: string;
  timestamp?: number;
  participant?: string;
  isFinal?: boolean;
}

// Theme type
export type Theme = 'light' | 'dark' | 'system';
