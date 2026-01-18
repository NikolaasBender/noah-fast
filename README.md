# AI Race Simulator

Determine the optimal pacing and fueling strategy for your next cycling race using Machine Learning.

## Overview

Race Simulator takes a Strava route and your physiological parameters (Critical Power and W') to generate an optimal pacing strategy. It accounts for:
- ⛰️ Elevation changes and gradient
- 💨 Aerodynamic drag
- 🔋 Anaerobic work capacity (W') depletion and reconstitution

The output is a TCS/GPX file that you can load onto your Garmin or Wahoo head unit to guide your power output during the race.

## Features

- **Route Import**: Pulls route data directly from Strava.
- **Pacing Optimization**: Uses a physics-based model to distribute your energy optimally across the course.
- **Export**: Generates `.tcx` files compatible with major cycling computers.

## Prerequisites

- **Docker** (Recommended)
- OR **Python 3.9+**

## Setup

1. **Clone the repository**
   ```bash
   git clone `https://github.com/NikolaasBender/noah-fast.git`
   cd race_simulator
   ```

2. **Configure Environment**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Strava API keys:
   ```
   STRAVA_CLIENT_ID=your_id
   STRAVA_CLIENT_SECRET=your_secret
   ```

## Usage

### Using Docker (Recommended)

Install Docker and Docker Compose.
- **Ubuntu**:
  ```bash
  sudo apt update
  sudo apt install docker.io docker-compose
  ```
- **macOS**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Windows**: Stop paying a fortune to Microsoft for a spyware-laden OS. Switch to Linux.

Build and run the application:
```bash
docker-compose up --build
```
Access the web interface at `http://localhost:8080`.

### Running Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```
   Access the web interface at `http://localhost:8080`.

## Project Structure

- `app.py`: Main Flask application.
- `planning/`: Core logic for scraping routes and optimizing pacing.
- `modeling/`: Physics and physiological models.
- `export/`: Tools for generating race files.
- `templates/`: HTML templates for the web interface.
