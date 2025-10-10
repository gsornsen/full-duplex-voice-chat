# Web Client User Guide

This guide explains how to use the Realtime Duplex Voice Demo web interface for voice conversations.

## Getting Started

### Prerequisites

Before using the web client, ensure:

1. Backend services are running (see [SETUP.md](./SETUP.md))
2. Web client is running on http://localhost:3000
3. Your browser has microphone permissions enabled
4. You're using a modern browser (Chrome, Firefox, Safari, or Edge)

### Accessing the Web Client

1. Open your web browser
2. Navigate to http://localhost:3000
3. You'll see the welcome screen with "Start voice chat" button

## Using the Voice Interface

### Starting a Conversation

1. Click the **"Start voice chat"** button on the welcome screen
2. Your browser will request microphone permission - click **"Allow"**
3. Wait for the connection to establish (you'll see a connecting indicator)
4. Once connected, you'll see the active session view with:
   - Agent status indicator
   - Audio visualization
   - Chat input (optional)
   - Control buttons

### Speaking with the Agent

The system supports **full-duplex** conversation, meaning you can speak naturally without waiting for the agent to finish:

1. **Just start speaking** - the system automatically detects your voice
2. The agent will respond with synthesized speech
3. You can **interrupt the agent** at any time (barge-in) - just start speaking and the agent will pause
4. The system handles turn-taking automatically based on voice activity detection

**Key Features:**

- **Voice Activity Detection (VAD)**: Automatically detects when you're speaking
- **Barge-in Support**: Interrupt the agent mid-sentence by speaking
- **Low Latency**: First audio response typically arrives in < 300ms
- **Continuous Conversation**: No need to click buttons between turns

### Using Text Chat (Optional)

In addition to voice, you can also send text messages:

1. Type your message in the chat input box at the bottom
2. Press **Enter** or click **Send**
3. The agent will speak the response
4. Text chat is useful for:
   - Testing without using your microphone
   - Sending complex or specific instructions
   - Quiet environments where you can't speak aloud

### Understanding the Interface

#### Agent Status Indicators

- **Connected**: Ready for conversation (green indicator)
- **Speaking**: Agent is currently speaking (blue animation)
- **Listening**: Waiting for your voice input
- **Disconnected**: Not connected (red indicator)

#### Audio Visualization

- Real-time audio level meter shows your microphone input
- Helps confirm your microphone is working
- Provides feedback that the system is receiving your voice

#### Control Buttons

- **Microphone**: Mute/unmute your microphone
- **Leave**: End the session and return to welcome screen
- **Settings** (if available): Adjust device settings

### Ending a Session

To end the conversation:

1. Click the **"Leave"** or **"End call"** button
2. The session will close and you'll return to the welcome screen
3. Click **"Start voice chat"** again to start a new session

## Best Practices

### For Optimal Voice Quality

1. **Use a good microphone**: Built-in laptop mics work, but headset mics are better
2. **Minimize background noise**: Find a quiet environment
3. **Speak clearly**: Normal conversational volume and pace
4. **Use headphones**: Prevents echo and feedback
5. **Stable internet**: For WebRTC, stable connection is more important than high bandwidth

### For Natural Conversations

1. **Speak naturally**: The system handles normal conversation flow
2. **Don't wait for prompts**: Start speaking whenever you want
3. **Interrupt when needed**: The barge-in feature makes it feel natural
4. **Use turn-taking cues**: Pause slightly at the end of your thought
5. **Be patient**: First response may take 200-400ms (first audio latency)

### For Testing

1. **Test microphone first**: Check audio visualization before speaking
2. **Start with simple queries**: "Hello", "How are you?", "What's the weather?"
3. **Try interruptions**: Start speaking while the agent is talking
4. **Use text for debugging**: Send text messages to rule out microphone issues

## Features Explained

### Barge-In (Interruption Handling)

The system's most notable feature is **barge-in support**:

**What it is**: The ability to interrupt the agent while it's speaking

**How it works**:
1. Agent starts speaking in response to your query
2. You start speaking (system detects voice activity)
3. Agent immediately pauses (< 50ms latency)
4. Your voice input is processed
5. Agent resumes or responds based on new input

**Why it matters**: Makes conversations feel more natural, like talking to a real person

**Example use case**:
- Agent: "The capital of France is Par—"
- You: "Actually, I meant Germany"
- Agent: *pauses immediately* "The capital of Germany is Berlin"

### Voice Activity Detection (VAD)

**What it is**: Automatic detection of when you're speaking vs. silence

**How it works**:
- Analyzes audio in real-time (20ms frames)
- Distinguishes speech from background noise
- Triggers state transitions (listening → speaking → responding)

**Benefits**:
- No need to hold push-to-talk buttons
- Hands-free operation
- Natural conversation flow

### Text-to-Speech (TTS) Models

The backend supports multiple TTS models with different characteristics:

- **Default**: Optimized for low latency (< 300ms first audio)
- **High Quality**: Better voice quality, slightly higher latency
- **Multi-lingual**: Supports multiple languages
- **Custom Voices**: LoRA-fine-tuned models for specific voice styles

**Note**: Model selection is configured on the backend; the web client uses whatever model the orchestrator routes to.

## Troubleshooting

### No Audio From Agent

**Symptoms**: You can speak but don't hear any response

**Solutions**:
1. Check browser audio isn't muted
2. Check system volume settings
3. Verify backend services are running:
   ```bash
   docker compose ps
   ```
4. Check orchestrator logs:
   ```bash
   docker compose logs -f orchestrator
   ```

### Microphone Not Working

**Symptoms**: Audio visualization shows no activity when you speak

**Solutions**:
1. Grant microphone permission when browser prompts
2. Check browser settings → Privacy → Microphone
3. Check system microphone isn't muted
4. Try a different browser
5. Check browser console for errors (F12 → Console tab)

### Poor Audio Quality

**Symptoms**: Choppy, distorted, or delayed audio

**Solutions**:
1. Check network connection stability
2. Close other bandwidth-intensive applications
3. Use wired headphones instead of Bluetooth
4. Restart the session
5. Check backend service health:
   ```bash
   docker compose logs -f
   ```

### Connection Failures

**Symptoms**: "Connection failed" or stuck on "Connecting..."

**Solutions**:
1. Verify LiveKit server is running:
   ```bash
   docker compose ps livekit
   ```
2. Check `.env.local` has correct `LIVEKIT_URL`
3. Test LiveKit health:
   ```bash
   curl http://localhost:7881/
   ```
4. Restart web client:
   ```bash
   # In src/client/web/
   pnpm dev
   ```
5. Restart backend services:
   ```bash
   docker compose restart
   ```

### High Latency

**Symptoms**: Long delays between your speech and agent response

**Expected latency**:
- First Audio Latency (FAL): < 300ms (p95)
- Barge-in latency: < 50ms (p95)

**If latency is higher**:
1. Check TTS worker is using GPU:
   ```bash
   docker compose logs tts0 | grep CUDA
   ```
2. Check system resource usage (CPU/GPU/RAM)
3. Verify no other processes are using the GPU
4. Check network latency (ping localhost)

## Advanced Features

### Developer Tools

Press **F12** in your browser to open developer tools:

- **Console**: View connection logs and errors
- **Network**: Monitor WebRTC connections
- **Performance**: Profile client-side performance

### Browser Compatibility

**Recommended browsers**:
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Required features**:
- WebRTC support
- Web Audio API
- getUserMedia API
- WebSocket support

### Session Persistence

- Sessions are temporary and don't persist across page reloads
- Refreshing the page ends the current session
- No conversation history is stored (by design)
- Each "Start voice chat" creates a new session

## Privacy & Security

### Data Handling

- **Audio processing**: All audio is processed in real-time and not stored
- **Conversation data**: No conversation logs are kept by default
- **Microphone access**: Only active during sessions; can be revoked in browser settings
- **Local deployment**: All services run on your local machine

### Security Considerations

For local development:
- Uses development keys (`devkey`/`secret`) - **DO NOT use in production**
- No encryption on local WebSocket connections
- No authentication required

For production deployment:
- Generate secure API keys: `docker run --rm livekit/livekit-server generate-keys`
- Use WSS (WebSocket Secure) instead of WS
- Implement authentication/authorization
- Enable TLS/HTTPS

## Performance Metrics

The system tracks several performance metrics:

### Latency Metrics

- **First Audio Latency (FAL)**: Time from your speech end to first audio frame
  - Target: < 300ms (p95)
  - Typical: 200-250ms

- **Barge-in Latency**: Time to pause agent when you start speaking
  - Target: < 50ms (p95)
  - Typical: 20-40ms

- **Frame Jitter**: Consistency of audio frame delivery
  - Target: < 10ms (p95)
  - Important for smooth playback

### Viewing Metrics

Metrics are logged in the browser console (F12 → Console):
- Connection latency
- Audio frame timing
- Session statistics

## Next Steps

- Explore different conversation topics
- Try the barge-in feature by interrupting the agent
- Test with both voice and text input
- Check out the [SETUP.md](./SETUP.md) for advanced configuration
- Review [Performance Tests](../../../tests/performance/README.md) for system capabilities

## Support & Resources

- **Project Documentation**: See `project_documentation/` directory
- **Integration Tests**: See `tests/integration/` for example usage
- **Performance Tests**: See `tests/performance/` for benchmarks
- **LiveKit Docs**: https://docs.livekit.io/
- **Report Issues**: Check project repository for issue tracker
