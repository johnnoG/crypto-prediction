import React from 'react';

export interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  retryLabel?: string;
  className?: string;
  showDetails?: boolean;
  error?: Error;
}

export function GenericError({
  title = "Something went wrong",
  message = "We encountered an unexpected error. Please try again.",
  onRetry,
  retryLabel = "Try Again",
  className = "",
  showDetails = false,
  error
}: ErrorStateProps) {
  return (
    <div className={`text-center py-12 ${className}`}>
      <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-100 flex items-center justify-center">
        <svg className="w-10 h-10 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      
      <h3 className="text-2xl font-black text-gray-900 mb-4">{title}</h3>
      <p className="text-gray-600 mb-8 max-w-md mx-auto leading-relaxed">{message}</p>
      
      {onRetry && (
        <button onClick={onRetry} className="coinbase-button">
          {retryLabel}
        </button>
      )}
      
      {showDetails && error && process.env.NODE_ENV === 'development' && (
        <details className="mt-8 text-left bg-gray-100 rounded-lg p-4 max-w-2xl mx-auto">
          <summary className="font-bold cursor-pointer mb-2 text-red-600">
            Error Details
          </summary>
          <pre className="text-sm text-gray-800 whitespace-pre-wrap break-words">
            {error.toString()}
          </pre>
        </details>
      )}
    </div>
  );
}

export function NetworkError({
  onRetry,
  className = ""
}: {
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div className={`text-center py-12 ${className}`}>
      <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-orange-100 flex items-center justify-center">
        <svg className="w-10 h-10 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
        </svg>
      </div>
      
      <h3 className="text-2xl font-black text-gray-900 mb-4">Connection Problem</h3>
      <p className="text-gray-600 mb-8 max-w-md mx-auto leading-relaxed">
        We're having trouble connecting to our servers. Please check your internet connection and try again.
      </p>
      
      <div className="space-y-4">
        {onRetry && (
          <button onClick={onRetry} className="coinbase-button mr-4">
            Retry Connection
          </button>
        )}
        
        <button 
          onClick={() => window.location.reload()}
          className="coinbase-button-secondary"
        >
          Refresh Page
        </button>
      </div>
      
      <div className="mt-8 text-sm text-gray-500">
        <p>Still having issues? Check your:</p>
        <ul className="mt-2 space-y-1">
          <li>• Internet connection</li>
          <li>• Firewall settings</li>
          <li>• VPN configuration</li>
        </ul>
      </div>
    </div>
  );
}

export function DataError({
  dataType = "data",
  onRetry,
  className = ""
}: {
  dataType?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div className={`text-center py-12 ${className}`}>
      <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-yellow-100 flex items-center justify-center">
        <svg className="w-10 h-10 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </div>
      
      <h3 className="text-2xl font-black text-gray-900 mb-4">No {dataType} Available</h3>
      <p className="text-gray-600 mb-8 max-w-md mx-auto leading-relaxed">
        We couldn't load the {dataType} you requested. This might be temporary.
      </p>
      
      {onRetry && (
        <button onClick={onRetry} className="coinbase-button">
          Reload {dataType}
        </button>
      )}
    </div>
  );
}

export function MaintenanceError({ 
  className = "" 
}: { 
  className?: string;
}) {
  return (
    <div className={`text-center py-12 ${className}`}>
      <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-blue-100 flex items-center justify-center">
        <svg className="w-10 h-10 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      
      <h3 className="text-2xl font-black text-gray-900 mb-4">Under Maintenance</h3>
      <p className="text-gray-600 mb-8 max-w-md mx-auto leading-relaxed">
        We're performing scheduled maintenance to improve your experience. We'll be back shortly.
      </p>
      
      <div className="text-sm text-gray-500">
        <p>Expected downtime: 5-10 minutes</p>
        <p className="mt-2">Thank you for your patience!</p>
      </div>
    </div>
  );
}

export function WebSocketError({
  onReconnect,
  isReconnecting = false,
  className = ""
}: {
  onReconnect?: () => void;
  isReconnecting?: boolean;
  className?: string;
}) {
  return (
    <div className={`market-card border-orange-200 ${className}`}>
      <div className="text-center py-12">
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-orange-100 flex items-center justify-center">
          <svg className="w-8 h-8 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        
        <h3 className="text-xl font-black text-gray-900 mb-4">Real-time Connection Lost</h3>
        <p className="text-gray-600 mb-6">
          {isReconnecting 
            ? "Attempting to reconnect to live data stream..."
            : "We're having trouble maintaining the live connection."
          }
        </p>
        
        {isReconnecting ? (
          <div className="flex justify-center">
            <div className="w-6 h-6 border-2 border-orange-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : (
          onReconnect && (
            <button onClick={onReconnect} className="coinbase-button">
              Reconnect
            </button>
          )
        )}
        
        <div className="mt-6 text-sm text-gray-500">
          <p>Don't worry - you're still seeing recent data from our cache</p>
        </div>
      </div>
    </div>
  );
}

// Hook for handling different error types
export function useErrorHandler() {
  const handleError = (error: unknown, context?: string) => {
    console.error(`Error in ${context || 'unknown context'}:`, error);
    
    // You could integrate with error reporting services here
    // e.g., Sentry, LogRocket, etc.
    
    if (error instanceof Error) {
      // Network errors
      if (error.name === 'NetworkError' || error.message.includes('fetch')) {
        return 'network';
      }
      
      // API errors
      if (error.message.includes('API') || error.message.includes('server')) {
        return 'api';
      }
      
      // WebSocket errors
      if (error.message.includes('WebSocket') || error.message.includes('connection')) {
        return 'websocket';
      }
    }
    
    return 'generic';
  };
  
  return { handleError };
}

