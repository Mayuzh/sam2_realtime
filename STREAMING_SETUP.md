# SAM2 Realtime Streaming Setup

This project publishes processed SAM2 output as production-oriented H.264/HLS streams. The public URLs are exposed with Cloudflare Tunnel and can be embedded in a frontend such as Vercel. Legacy MJPEG endpoints remain available for debugging.

## Stream Map

Each script starts its own local HTTP stream server:

| Script | Local URL | Public URL |
| --- | --- | --- |
| `notebooks/sam2_realtime.py` | `http://localhost:8000/hls/stream.m3u8` | `https://santa-cruz.realtimeshorelinestream.store/hls/stream.m3u8` |
| `notebooks/sam2_realtime2.py` | `http://localhost:8001/hls/stream.m3u8` | `https://jennette.realtimeshorelinestream.store/hls/stream.m3u8` |
| `notebooks/sam2_realtime3.py` | `http://localhost:8002/hls/stream.m3u8` | `https://point-reyes.realtimeshorelinestream.store/hls/stream.m3u8` |

Health checks are available at `/health`, for example:

```text
http://localhost:8000/health
```

The scripts run headless, so stop them with `Ctrl+C`.

FFmpeg is required for H.264 encoding. Install FFmpeg on each streaming computer and make sure `ffmpeg` is on `PATH`, or install the bundled Python fallback in the environment used to run the scripts:

```powershell
pip install imageio-ffmpeg
```

## Tunnel Config Files

There are three Cloudflare Tunnel config files:

| File | Use Case | Routes |
| --- | --- | --- |
| `cloudflared-sam2-cameras.yml` | Single computer running all three streams | `santa-cruz -> 8000`, `jennette -> 8001`, `point-reyes -> 8002` |
| `cloudflared-sam2-cameras-a.yml` | Computer A in the two-computer setup | `santa-cruz -> 8000` |
| `cloudflared-sam2-cameras-b.yml` | Computer B in the two-computer setup | `jennette -> 8001`, `point-reyes -> 8002` |

Use exactly one config per tunnel process.

## Single-Computer Setup

Use this when one computer runs all three SAM2 scripts.

Start the three local streams in three terminals:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime.py
```

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime2.py
```

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime3.py
```

Start the all-in-one tunnel:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime
cloudflared tunnel --config cloudflared-sam2-cameras.yml run sam2-cameras
```

## Two-Computer Setup

Use this when Computer A runs `sam2_realtime.py`, and Computer B runs `sam2_realtime2.py` and `sam2_realtime3.py`.

### Computer A

Computer A maps:

```text
santa-cruz.realtimeshorelinestream.store -> http://localhost:8000
```

Start the local stream:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime.py
```

Start the Computer A tunnel:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime
cloudflared tunnel --config cloudflared-sam2-cameras-a.yml run sam2-cameras
```

### Computer B

Computer B maps:

```text
jennette.realtimeshorelinestream.store -> http://localhost:8001
point-reyes.realtimeshorelinestream.store -> http://localhost:8002
```

Log into the Cloudflare account that owns `realtimeshorelinestream.store`:

```powershell
cloudflared tunnel login
```

Create the Computer B tunnel if it does not already exist:

```powershell
cloudflared tunnel create sam2-cameras-b
```

Get the tunnel ID:

```powershell
cloudflared tunnel list
```

Edit `cloudflared-sam2-cameras-b.yml` and replace `<SECONDARY_TUNNEL_ID>` with the ID for `sam2-cameras-b`.

Route the Computer B hostnames to the Computer B tunnel:

```powershell
cloudflared tunnel route dns sam2-cameras-b jennette.realtimeshorelinestream.store
cloudflared tunnel route dns sam2-cameras-b point-reyes.realtimeshorelinestream.store
```

Start the local streams in two terminals:

```powershell
cd C:\Users\monaz\Documents\GitHub\sam2_realtime\notebooks
python sam2_realtime2.py
```

```powershell
cd C:\Users\monaz\Documents\GitHub\sam2_realtime\notebooks
python sam2_realtime3.py
```

Start the Computer B tunnel:

```powershell
cd C:\Users\monaz\Documents\GitHub\sam2_realtime
cloudflared tunnel --config cloudflared-sam2-cameras-b.yml run sam2-cameras-b
```

## Credentials

Only the computer running a Cloudflare tunnel needs credentials. Website viewers do not need Cloudflare credentials.

By default, `cloudflared` stores tunnel credentials here:

```text
C:\Users\<windows-user>\.cloudflared\<TUNNEL_ID>.json
```

## Frontend Usage

Use these HLS playlist URLs in the frontend:

```text
"https://santa-cruz.realtimeshorelinestream.store/hls/stream.m3u8",
"https://jennette.realtimeshorelinestream.store/hls/stream.m3u8",
"https://point-reyes.realtimeshorelinestream.store/hls/stream.m3u8",
```

Safari can play an HLS URL directly in a `<video>` element. Chrome, Firefox, and Edge should use `hls.js`:

```javascript
import Hls from "hls.js";

const video = document.querySelector("video");
const streamUrl =
  "https://santa-cruz.realtimeshorelinestream.store/hls/stream.m3u8";

if (video.canPlayType("application/vnd.apple.mpegurl")) {
  video.src = streamUrl;
} else if (Hls.isSupported()) {
  const hls = new Hls({
    liveSyncDurationCount: 3,
    liveMaxLatencyDurationCount: 6,
  });
  hls.loadSource(streamUrl);
  hls.attachMedia(video);
}
```

The element should be configured for autoplay-safe live playback:

```html
<video autoplay muted playsinline controls></video>
```
