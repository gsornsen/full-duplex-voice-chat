import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// App configuration utilities
export async function getAppConfig(headers: Headers) {
  const host = headers.get("host") || "localhost:3000"
  return {
    serverUrl: process.env.NEXT_PUBLIC_LIVEKIT_URL || `wss://${host}`,
    apiKey: process.env.NEXT_PUBLIC_LIVEKIT_API_KEY || "devkey"
  }
}
