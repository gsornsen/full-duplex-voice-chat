'use client';

import { useEffect, useMemo, useState } from 'react';
import { Room, RoomEvent, Track } from 'livekit-client';
import { motion } from 'motion/react';
import { RoomAudioRenderer, RoomContext, StartAudio } from '@livekit/components-react';
import { toastAlert } from '@/components/alert-toast';
import { SessionView } from '@/components/session-view';
import { Toaster } from '@/components/ui/sonner';
import { Welcome } from '@/components/welcome';
import useConnectionDetails from '@/hooks/useConnectionDetails';
import type { AppConfig } from '@/lib/types';

const MotionWelcome = motion.create(Welcome);
const MotionSessionView = motion.create(SessionView);

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  // Create Room with audio capture defaults (M10 fix - apply AGC constraints at Room level)
  const room = useMemo(
    () =>
      new Room({
        adaptiveStream: true,
        dynacast: true,
        // Audio defaults applied to all local audio tracks
        audioCaptureDefaults: {
          autoGainControl: false, // Keep disabled for natural dynamics
          echoCancellation: true, // ENABLE to prevent feedback loops
          noiseSuppression: true, // ENABLE to filter background noise
          channelCount: 1, // Mono audio
          sampleRate: 48000, // Match server expectation
        },
      }),
    []
  );
  const [sessionStarted, setSessionStarted] = useState(false);
  const { refreshConnectionDetails, existingOrRefreshConnectionDetails } =
    useConnectionDetails(appConfig);

  useEffect(() => {
    const onDisconnected = () => {
      setSessionStarted(false);
      refreshConnectionDetails();
    };
    const onMediaDevicesError = (error: Error) => {
      toastAlert({
        title: 'Encountered an error with your media devices',
        description: `${error.name}: ${error.message}`,
      });
    };
    const onReconnected = async () => {
      // After reconnection, just verify audio settings
      // DO NOT manually enable/disable microphone - LiveKit handles this
      console.log('[AGC Debug] Reconnection detected, verifying audio settings');
      try {
        // Verify actual constraints after reconnection
        const audioTrack = room.localParticipant.getTrackPublication(Track.Source.Microphone);
        if (audioTrack?.track) {
          const settings = audioTrack.track.mediaStreamTrack.getSettings();
          console.log('[AGC Debug] MediaStreamTrack settings after reconnection:', {
            autoGainControl: settings.autoGainControl,
            echoCancellation: settings.echoCancellation,
            noiseSuppression: settings.noiseSuppression,
            sampleRate: settings.sampleRate,
            channelCount: settings.channelCount,
          });
        } else {
          console.warn('[AGC Debug] No audio track found after reconnection');
        }
      } catch (error) {
        console.error('Failed to verify microphone after reconnection:', error);
      }
    };
    const onLocalTrackPublished = (publication: {
      kind: string;
      trackSid?: string;
      track?: { mediaStreamTrack: MediaStreamTrack };
    }) => {
      // Verify AGC settings when local tracks are published
      if (publication.kind === 'audio' && publication.track) {
        const settings = publication.track.mediaStreamTrack.getSettings();
        console.log('[AGC Debug] Local audio track published with settings:', {
          trackId: publication.trackSid,
          autoGainControl: settings.autoGainControl,
          echoCancellation: settings.echoCancellation,
          noiseSuppression: settings.noiseSuppression,
          sampleRate: settings.sampleRate,
          channelCount: settings.channelCount,
        });

        // Warn if AGC is still enabled
        if (settings.autoGainControl) {
          console.warn('[AGC Debug] WARNING: AGC is enabled despite audioCaptureDefaults!');
        }
      }
    };
    room.on(RoomEvent.MediaDevicesError, onMediaDevicesError);
    room.on(RoomEvent.Disconnected, onDisconnected);
    room.on(RoomEvent.Reconnected, onReconnected);
    room.on(RoomEvent.LocalTrackPublished, onLocalTrackPublished);
    return () => {
      room.off(RoomEvent.Disconnected, onDisconnected);
      room.off(RoomEvent.MediaDevicesError, onMediaDevicesError);
      room.off(RoomEvent.Reconnected, onReconnected);
      room.off(RoomEvent.LocalTrackPublished, onLocalTrackPublished);
    };
  }, [room, refreshConnectionDetails]);

  useEffect(() => {
    let aborted = false;
    if (sessionStarted) {
      console.log('[AGC Debug] Session start requested, current room state:', room.state);

      // Ensure room is disconnected before connecting
      // This handles race conditions where disconnect() hasn't completed
      (async () => {
        try {
          // If room is not disconnected, disconnect it first
          if (room.state !== 'disconnected') {
            console.log('[AGC Debug] Room not disconnected, disconnecting first...');
            await room.disconnect();
            console.log('[AGC Debug] Room disconnected, ready to connect');
          }

          // Get connection details
          const connectionDetails = await existingOrRefreshConnectionDetails();

          console.log('[AGC Debug] Connecting to room with audio enabled');

          // Connect to room (audio will be enabled via local tracks)
          await room.connect(connectionDetails.serverUrl, connectionDetails.participantToken, {
            autoSubscribe: true,
          });

          console.log('[AGC Debug] Connected to room successfully');

          // NOTE: Audio track settings verification happens in LocalTrackPublished event
          // Don't check here - track isn't published yet at this point in the lifecycle
        } catch (error) {
          if (aborted) {
            // Once the effect has cleaned up after itself, drop any errors
            //
            // These errors are likely caused by this effect rerunning rapidly,
            // resulting in a previous run `disconnect` running in parallel with
            // a current run `connect`
            return;
          }

          toastAlert({
            title: 'There was an error connecting to the agent',
            description: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
          });
        }
      })();
    }
    return () => {
      aborted = true;
      room.disconnect();
    };
  }, [room, sessionStarted, existingOrRefreshConnectionDetails]);

  const { startButtonText } = appConfig;

  return (
    <main>
      <MotionWelcome
        key="welcome"
        startButtonText={startButtonText}
        onStartCall={() => setSessionStarted(true)}
        disabled={sessionStarted}
        initial={{ opacity: 1 }}
        animate={{ opacity: sessionStarted ? 0 : 1 }}
        transition={{ duration: 0.5, ease: 'linear', delay: sessionStarted ? 0 : 0.5 }}
      />

      <RoomContext.Provider value={room}>
        <RoomAudioRenderer />
        <StartAudio label="Start Audio" />
        {/* --- */}
        <MotionSessionView
          key="session-view"
          appConfig={appConfig}
          disabled={!sessionStarted}
          sessionStarted={sessionStarted}
          initial={{ opacity: 0 }}
          animate={{ opacity: sessionStarted ? 1 : 0 }}
          transition={{
            duration: 0.5,
            ease: 'linear',
            delay: sessionStarted ? 0.5 : 0,
          }}
        />
      </RoomContext.Provider>

      <Toaster />
    </main>
  );
}
