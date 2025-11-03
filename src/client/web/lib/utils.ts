import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Utility function to merge Tailwind CSS classes with clsx
 * Standard Shadcn UI utility
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Theme storage key for localStorage
 */
export const THEME_STORAGE_KEY = 'vite-ui-theme';

/**
 * Media query for dark mode preference
 */
export const THEME_MEDIA_QUERY = '(prefers-color-scheme: dark)';

/**
 * Get application configuration
 */
export async function getAppConfig(): Promise<import('./types').AppConfig> {
  // Import AppConfig defaults from app-config.ts
  // For now, return a default config matching AppConfig interface
  return {
    companyName: 'Realtime Duplex Voice Demo',
    pageTitle: 'Voice Chat',
    pageDescription: 'Full-duplex voice conversation',
    supportsChatInput: true,
    supportsVideoInput: false,
    supportsScreenShare: false,
    isPreConnectBufferEnabled: true,
    logo: '/lk-logo.svg',
    accent: '#002cf2',
    logoDark: '/lk-logo-dark.svg',
    accentDark: '#1fd5f9',
    startButtonText: 'Start voice chat',
    agentName: 'voice-assistant',
  };
}

/**
 * Convert transcription data to chat message format
 */
export function transcriptionToChatMessage(transcription: {
  text: string;
  timestamp?: number;
  participant?: string | { name: string };
  isFinal?: boolean;
}): import('./types').ChatMessage {
  return {
    id: `trans-${transcription.timestamp || Date.now()}`,
    message: transcription.text,
    timestamp: transcription.timestamp || Date.now(),
    name:
      typeof transcription.participant === 'object' && transcription.participant !== null
        ? transcription.participant.name
        : transcription.participant || 'User',
    isSelf: true,
    isFinal: transcription.isFinal ?? true,
  };
}
