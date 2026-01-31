import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from './lib/queryClient';
import Dashboard from './components/Dashboard';
import OAuthCallback from './components/auth/OAuthCallback';
import ErrorBoundary from './components/ErrorBoundary';
import { AuthProvider } from './contexts/AuthContext';
import { useEffect } from 'react';

function App() {
  useEffect(() => {
    // Always use dark mode for this professional trading dashboard
    document.documentElement.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  }, []);

  const isOAuthCallback = window.location.pathname.startsWith('/oauth/callback/');

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <div className="min-h-screen bg-background text-foreground">
            {isOAuthCallback ? <OAuthCallback /> : <Dashboard />}
          </div>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
