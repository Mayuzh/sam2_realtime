# SAM2 Realtime Streaming Setup

## Local Stream URLs

Each script starts its own HTTP stream server:

| Script                        | Local URL                     | Public Host                                               |
| ----------------------------- | ----------------------------- | --------------------------------------------------------- |
| `notebooks/sam2_realtime.py`  | `http://localhost:8000/video` | `https://santa-cruz.realtimeshorelinestream.store/video`  |
| `notebooks/sam2_realtime2.py` | `http://localhost:8001/video` | `https://jennette.realtimeshorelinestream.store/video`    |
| `notebooks/sam2_realtime3.py` | `http://localhost:8002/video` | `https://point-reyes.realtimeshorelinestream.store/video` |

Health checks are available at `/health`, for example:

```text
http://localhost:8000/health
```

## Start The Three Streams

Open three terminals and run one script in each.

Terminal 1:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime.py
```

Terminal 2:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime2.py
```

Terminal 3:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime\notebooks
python sam2_realtime3.py
```

The scripts run headless, so stop them with `Ctrl+C`.

## Start Cloudflare Tunnel

After the local streams are running, start the tunnel from the repo root:

```powershell
cd C:\Users\Jeff\Desktop\Coding\sam2_realtime
cloudflared tunnel --config cloudflared-sam2-cameras.yml run sam2-cameras
```

The tunnel config is in:

```text
cloudflared-sam2-cameras.yml
```

It maps the three public hostnames to local ports `8000`, `8001`, and `8002`.
