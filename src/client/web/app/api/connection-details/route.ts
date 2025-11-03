import { NextResponse } from 'next/server';
import { AccessToken, type AccessTokenOptions, type VideoGrant } from 'livekit-server-sdk';
import { RoomConfiguration } from '@livekit/protocol';

// NOTE: you are expected to define the following environment variables in `.env.local`:
// - LIVEKIT_API_KEY, LIVEKIT_API_SECRET (server-side, for token generation)
// - LIVEKIT_URL (optional, server-side internal URL - not currently used but available for future use)
// - NEXT_PUBLIC_LIVEKIT_URL (optional, client-side, public WSS URL - falls back to auto-detection)
const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
// Server-side internal URL (currently unused, kept for reference)
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const LIVEKIT_URL = process.env.LIVEKIT_URL;
const PUBLIC_LIVEKIT_URL = process.env.NEXT_PUBLIC_LIVEKIT_URL;

/**
 * Derives the WSS URL for the client based on the request origin.
 * Supports multiple deployment modes:
 * - localhost (bare metal)
 * - voicechat.local (with /etc/hosts)
 * - Custom hostnames/IPs
 *
 * If accessed via HTTPS on port 8443, returns WSS on port 8444 on the same hostname.
 * Falls back to NEXT_PUBLIC_LIVEKIT_URL if origin cannot be determined.
 */
function getClientWssUrl(req: Request): string {
  // Prefer explicit environment variable if set
  if (PUBLIC_LIVEKIT_URL) {
    return PUBLIC_LIVEKIT_URL;
  }

  // Try to derive from request origin
  const origin = req.headers.get('origin') || req.headers.get('referer');
  if (origin) {
    try {
      const url = new URL(origin);
      const hostname = url.hostname;
      const port = url.port ? parseInt(url.port, 10) : url.protocol === 'https:' ? 443 : 80;

      // Convert HTTPS (8443) to WSS (8444) on the same hostname
      // Supports: localhost, voicechat.local, IP addresses, etc.
      const wssPort = port === 8443 ? 8444 : port === 443 ? 8444 : 7880;
      const wssUrl = `wss://${hostname}:${wssPort}`;

      return wssUrl;
    } catch {
      // Invalid URL, fall through to default
    }
  }

  // Try to get from Host header as fallback
  const host = req.headers.get('host');
  if (host) {
    try {
      const [hostname, portStr] = host.split(':');
      const port = portStr ? parseInt(portStr, 10) : 443;
      const wssPort = port === 8443 ? 8444 : port === 443 ? 8444 : 7880;
      return `wss://${hostname}:${wssPort}`;
    } catch {
      // Invalid host, fall through to error
    }
  }

  // No fallback available - must set NEXT_PUBLIC_LIVEKIT_URL
  throw new Error(
    'Cannot determine WSS URL. Set NEXT_PUBLIC_LIVEKIT_URL environment variable or ensure request includes Origin/Referer header.'
  );
}

// don't cache the results
export const revalidate = 0;

export type ConnectionDetails = {
  serverUrl: string;
  roomName: string;
  participantName: string;
  participantToken: string;
};

export async function POST(req: Request) {
  try {
    // Determine client WSS URL (supports multiple deployment modes)
    const clientWssUrl = getClientWssUrl(req);

    if (API_KEY === undefined) {
      throw new Error('LIVEKIT_API_KEY is not defined');
    }
    if (API_SECRET === undefined) {
      throw new Error('LIVEKIT_API_SECRET is not defined');
    }

    // Parse agent configuration from request body
    const body = await req.json();
    const agentName: string = body?.room_config?.agents?.[0]?.agent_name;

    // Generate participant token
    const participantName = 'user';
    const participantIdentity = `voice_assistant_user_${Math.floor(Math.random() * 10_000)}`;
    const roomName = `voice_assistant_room_${Math.floor(Math.random() * 10_000)}`;

    const participantToken = await createParticipantToken(
      { identity: participantIdentity, name: participantName },
      roomName,
      agentName
    );

    // Return connection details
    // Use auto-detected or configured WSS URL for client connection
    // The client connects from the browser, so it needs the public URL accessible from the host machine
    const data: ConnectionDetails = {
      serverUrl: clientWssUrl,
      roomName,
      participantToken: participantToken,
      participantName,
    };
    const headers = new Headers({
      'Cache-Control': 'no-store',
    });
    return NextResponse.json(data, { headers });
  } catch (error) {
    if (error instanceof Error) {
      console.error(error);
      return NextResponse.json({ error: error.message }, { status: 500 });
    }
    return NextResponse.json({ error: 'Unknown error occurred' }, { status: 500 });
  }
}

function createParticipantToken(
  userInfo: AccessTokenOptions,
  roomName: string,
  agentName?: string
): Promise<string> {
  const at = new AccessToken(API_KEY, API_SECRET, {
    ...userInfo,
    ttl: '15m',
  });
  const grant: VideoGrant = {
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true,
  };
  at.addGrant(grant);

  if (agentName) {
    at.roomConfig = new RoomConfiguration({
      agents: [{ agentName }],
    });
  }

  return at.toJwt();
}
