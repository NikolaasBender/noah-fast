# Required Environment Variables

When deploying to Railway (or any other provider), you must configure the following environment variables in your project settings:

| Variable | Description |
| :--- | :--- |
| `STRAVA_CLIENT_ID` | Your Strava Application Client ID |
| `STRAVA_CLIENT_SECRET` | Your Strava Application Client Secret |
| `FLASK_SECRET_KEY` | A long, random string used to sign session cookies (e.g., generated with `openssl rand -hex 32`) |
| `PORT` | (Automatically set by Railway) The port your app binds to |

> [!IMPORTANT]
> Ensure `FLASK_ENV` is set to `production` in your deployment environment to enable security features like `Secure` and `HttpOnly` flags for cookies.
