# Setup

winget install cloudflare.cloudflared
cloudflared login

# Run (in different terminals)

cd notebooks
conda activate sam2
python .\sam2_realtime.py
python .\sam2_realtime2.py
python .\sam2_realtime3.py

# In a different terminal...

cloudflared tunnel run sam2-cameras

# Endpoints:

- **Walton Lighthouse**: https://walton.realtimeshorelinestream.store/stream
- **Jennette North**: https://jennette.realtimeshorelinestream.store/stream
- **TMMC PRLS**: https://tmmc.realtimeshorelinestream.store/stream
