> [!WARNING]
> This is heavily vibe coded

# 'AI' Race Simulator

Determine the optimal pacing and fueling strategy for your next cycling race using Physics & Physiology.

## Overview

Race Simulator takes a Strava route and your physiological parameters (Critical Power and W') to generate an optimal pacing strategy. It accounts for:
- ⛰️ Elevation changes and gradient
- 💨 Aerodynamic drag
- 🔋 Anaerobic work capacity (W') depletion and reconstitution

The output is a TCX file that you can load onto your Garmin or Wahoo head unit to guide your power output during the race.

## Features

- **Route Import**: Pulls route data directly from Strava.
- **Pacing Optimization**: Uses a physics-based model to distribute your energy optimally across the course.
- **Stateless Architecture**: Secure and privacy-focused; no user data is persisted on disk. Session data lives in encrypted cookies.
- **Auto-Export**: Generates `.tcx` files compatible with major cycling computers.

## Prerequisites

- **Docker** (Recommended)
- OR **Python 3.10+**

## Quick Start (Docker)

1. **Clone the repository**
   ```bash
   git clone https://github.com/NikolaasBender/noah-fast.git
   cd race_simulator
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and add STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET
   ```

3. **Run**
   ```bash
   docker compose up --build
   ```
   Access the web interface at `http://localhost:8080`.

## Deployment

### Railway / Cloud
This project is configured for easy deployment on platforms like Railway. 

1. **Environment Variables**: Refer to [DEPLOYMENT_VARS.md](DEPLOYMENT_VARS.md) for the required configuration.
2. **Docker**: The `Dockerfile` is production-ready and supports dynamic port binding.
3. **CI/CD**: A GitHub Action (`.github/workflows/publish.yml`) is included to automatically publish the Docker image to GHCR on pushes to `main`.

## Development

### Local Setup (No Docker)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run**
   ```bash
   # Linux/Mac
   export FLASK_APP=app.py
   flask run --port=8080
   ```

## Project Structure

- `app.py`: Main Flask application.
- `planning/`: Core logic for scraping routes and optimizing pacing.
- `modeling/`: Physics and physiological models.
- `export/`: Tools for generating race files.
- `templates/`: HTML templates for the web interface.
