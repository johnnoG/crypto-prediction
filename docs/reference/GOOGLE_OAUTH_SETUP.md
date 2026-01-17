# 🔐 Google OAuth Setup Guide

This guide will help you set up Google OAuth for social authentication in your CryptoForecast application.

## 📋 Prerequisites

- Google account
- Access to Google Cloud Console
- Your application running locally

## 🚀 Step-by-Step Setup

### Step 1: Create Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project" or select an existing project
3. Give your project a name (e.g., "CryptoForecast")
4. Click "Create"

### Step 2: Enable Google APIs

1. In the Google Cloud Console, navigate to "APIs & Services" > "Library"
2. Search for and enable these APIs:
   - **Google+ API** (for user profile information)
   - **Google Identity API** (for OAuth 2.0)

### Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" > "OAuth consent screen"
2. Choose "External" user type (unless you have a Google Workspace)
3. Fill in the required information:
   - **App name**: CryptoForecast
   - **User support email**: Your email
   - **Developer contact information**: Your email
4. Add these scopes (if asked):
   - `../auth/userinfo.email`
   - `../auth/userinfo.profile`
   - `openid`
5. Save and continue

### Step 4: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Web application"
4. Configure the settings:
   - **Name**: CryptoForecast Web Client
   - **Authorized JavaScript origins**:
     - `http://127.0.0.1:5173`
     - `http://localhost:5173`
   - **Authorized redirect URIs**:
     - `http://127.0.0.1:8000/api/auth/oauth/google/callback`
     - `http://localhost:8000/api/auth/oauth/google/callback`
5. Click "Create"
6. **Important**: Copy the Client ID and Client Secret

### Step 5: Configure Environment Variables

Create a `.env` file in your project root with the following content:

```env
# Authentication
SECRET_KEY=your-super-secret-key-change-in-production-min-32-chars
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/api/auth/oauth/google/callback

# Database
DATABASE_URL=sqlite:///./dev.db

# CORS
ALLOWED_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

**Replace the placeholder values with your actual Google OAuth credentials.**

### Step 6: Test the Integration

1. Start your application:
   ```powershell
   python main.py
   ```

2. Open your browser to `http://127.0.0.1:5173`

3. Click "Sign In" or "Sign Up"

4. Click the "Continue with Google" button

5. You should be redirected to Google's login page

6. After successful authentication, you'll be redirected back to your app

## 🔧 Troubleshooting

### Common Issues:

1. **"redirect_uri_mismatch" error**:
   - Make sure the redirect URI in Google Console exactly matches your configuration
   - Check for trailing slashes or http vs https

2. **"invalid_client" error**:
   - Verify your Client ID and Client Secret are correct
   - Make sure there are no extra spaces in your `.env` file

3. **"access_denied" error**:
   - The user cancelled the OAuth flow
   - Check your OAuth consent screen configuration

4. **CORS errors**:
   - Make sure your frontend URL is in the `ALLOWED_ORIGINS` environment variable

### Testing OAuth Endpoints:

You can test the OAuth flow manually:

1. **Get authorization URL**:
   ```
   GET http://127.0.0.1:8000/api/auth/oauth/google
   ```

2. **Check API documentation**:
   ```
   http://127.0.0.1:8000/docs
   ```

## 🎉 Success!

Once configured correctly, users will be able to:

- Sign in with their Google account
- Have their profile automatically created
- Access all authenticated features
- Sign out and sign back in seamlessly

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Use strong, unique secret keys in production
- Consider implementing additional security measures like rate limiting
- Regularly rotate your OAuth credentials

## 📚 Additional Resources

- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [FastAPI OAuth Documentation](https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/)
- [React OAuth Best Practices](https://auth0.com/blog/react-authentication-with-oauth2/)

---

**Need help?** The Google OAuth integration is now fully implemented in your codebase with proper error handling and user experience!
