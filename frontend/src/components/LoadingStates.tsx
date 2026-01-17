import React from 'react';

// Enhanced skeleton components with better animations and variety

export function SkeletonCard({ className = "" }: { className?: string }) {
  return (
    <div className={`animate-pulse ${className}`}>
      <div className="bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200 bg-[length:200%_100%] animate-shimmer rounded-lg h-full"></div>
    </div>
  );
}

export function SkeletonText({ 
  lines = 1, 
  className = "",
  widths = ["w-full"] 
}: { 
  lines?: number;
  className?: string;
  widths?: string[];
}) {
  return (
    <div className={`animate-pulse space-y-3 ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <div 
          key={i}
          className={`h-4 bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200 bg-[length:200%_100%] animate-shimmer rounded ${widths[i % widths.length]}`}
        />
      ))}
    </div>
  );
}

export function CryptoTableSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="crypto-table">
      <div className="crypto-table-header">
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-4 text-sm font-black text-gray-600 uppercase tracking-wider">Name</div>
          <div className="col-span-2 text-sm font-black text-gray-600 uppercase tracking-wider text-right">Price</div>
          <div className="col-span-2 text-sm font-black text-gray-600 uppercase tracking-wider text-right">24h Change</div>
          <div className="col-span-2 text-sm font-black text-gray-600 uppercase tracking-wider text-right">Market Cap</div>
          <div className="col-span-2 text-sm font-black text-gray-600 uppercase tracking-wider text-right">Volume (24h)</div>
        </div>
      </div>
      <div className="divide-y divide-gray-100">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="px-8 py-8">
            <div className="grid grid-cols-12 gap-6 items-center">
              <div className="col-span-4 flex items-center space-x-4">
                <div className="flex items-center space-x-3">
                  <SkeletonText lines={1} widths={["w-8"]} />
                </div>
                <SkeletonCard className="w-12 h-12 rounded-2xl" />
                <div className="space-y-3">
                  <SkeletonText lines={1} widths={["w-28"]} />
                  <SkeletonText lines={1} widths={["w-20"]} />
                </div>
              </div>
              <div className="col-span-2 text-right">
                <SkeletonText lines={1} widths={["w-24"]} />
              </div>
              <div className="col-span-2 text-right">
                <SkeletonText lines={1} widths={["w-20"]} />
              </div>
              <div className="col-span-2 text-right">
                <SkeletonText lines={1} widths={["w-28"]} />
              </div>
              <div className="col-span-2 text-right">
                <SkeletonText lines={1} widths={["w-24"]} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PriceTickerSkeleton() {
  return (
    <div className="bg-gradient-to-r from-gray-900 via-black to-gray-900 text-white py-4 overflow-hidden">
      <div className="flex animate-scroll-x space-x-16">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="flex items-center space-x-4 whitespace-nowrap">
            <SkeletonCard className="w-8 h-8 rounded-full" />
            <SkeletonText lines={1} widths={["w-16"]} className="text-white" />
            <SkeletonText lines={1} widths={["w-20"]} className="text-white" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function StatusBarSkeleton() {
  return (
    <div className="bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-10">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <SkeletonCard className="w-4 h-4 rounded-full" />
                <SkeletonText lines={1} widths={["w-24"]} />
              </div>
            ))}
          </div>
          <SkeletonText lines={1} widths={["w-32"]} />
        </div>
      </div>
    </div>
  );
}

export function LoadingSpinner({ 
  size = "md", 
  className = "",
  message = "Loading..."
}: { 
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
  message?: string;
}) {
  const sizeClasses = {
    sm: "w-6 h-6",
    md: "w-8 h-8", 
    lg: "w-12 h-12",
    xl: "w-16 h-16"
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-4 ${className}`}>
      <div className={`${sizeClasses[size]} relative`}>
        <div className="absolute inset-0 rounded-full border-4 border-gray-200"></div>
        <div className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent animate-spin"></div>
      </div>
      {message && (
        <p className="text-gray-600 font-medium animate-pulse">{message}</p>
      )}
    </div>
  );
}

export function FullPageLoading({ message = "Loading your crypto dashboard..." }: { message?: string }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-blue-50 flex items-center justify-center">
      <div className="text-center">
        <div className="w-24 h-24 mx-auto mb-8 relative">
          <div className="absolute inset-0 rounded-full border-8 border-gray-200"></div>
          <div className="absolute inset-0 rounded-full border-8 border-blue-500 border-t-transparent animate-spin"></div>
          <div className="absolute inset-4 rounded-full border-4 border-purple-300 border-b-transparent animate-spin-reverse"></div>
        </div>
        
        <h2 className="text-3xl font-black text-gray-900 mb-4">
          CryptoForecast
        </h2>
        
        <p className="text-xl text-gray-600 mb-8 animate-pulse">
          {message}
        </p>
        
        <div className="flex justify-center space-x-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div 
              key={i}
              className="w-3 h-3 bg-blue-500 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 0.2}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

// Progressive loading component that shows different states
export function ProgressiveLoader({ 
  stages = ["Connecting to API...", "Fetching market data...", "Processing results..."],
  currentStage = 0,
  className = ""
}: {
  stages?: string[];
  currentStage?: number;
  className?: string;
}) {
  return (
    <div className={`space-y-6 ${className}`}>
      <LoadingSpinner size="lg" message={stages[currentStage]} />
      
      <div className="w-full max-w-md mx-auto">
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>Progress</span>
          <span>{Math.round(((currentStage + 1) / stages.length) * 100)}%</span>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
          />
        </div>
        
        <div className="mt-4 space-y-2">
          {stages.map((stage, index) => (
            <div key={index} className="flex items-center space-x-3 text-sm">
              <div className={`w-4 h-4 rounded-full flex items-center justify-center ${
                index < currentStage 
                  ? 'bg-green-500' 
                  : index === currentStage 
                  ? 'bg-blue-500 animate-pulse' 
                  : 'bg-gray-300'
              }`}>
                {index < currentStage && (
                  <svg className="w-2 h-2 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
              <span className={index <= currentStage ? 'text-gray-900' : 'text-gray-500'}>
                {stage}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

