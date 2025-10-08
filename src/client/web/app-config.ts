import type { AppConfig } from './lib/types';

export const APP_CONFIG_DEFAULTS: AppConfig = {
  companyName: 'Realtime Duplex Voice Demo',
  pageTitle: 'Realtime Duplex Voice Chat',
  pageDescription: 'Full-duplex voice conversation with barge-in support and hot-swappable TTS models',

  supportsChatInput: true,
  supportsVideoInput: false,  // Voice-only for now
  supportsScreenShare: false,  // Voice-only for now
  isPreConnectBufferEnabled: true,

  logo: '/lk-logo.svg',
  accent: '#002cf2',
  logoDark: '/lk-logo-dark.svg',
  accentDark: '#1fd5f9',
  startButtonText: 'Start voice chat',

  agentName: 'voice-assistant',  // Default agent name for our orchestrator
};
