import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';


function NewsPanel() {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  
  // Real-time news from our backend
  const { data: newsData, isLoading, error, refetch } = useQuery({
    queryKey: ['crypto-news'],
    queryFn: () => api.getNews({ limit: 24, realtime: true }),
    refetchInterval: 120000, // Refresh every 2 minutes
    staleTime: 60000, // Consider data stale after 1 minute
  });

  // Helper functions
  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  const getSentimentStyles = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive': return {
        bg: 'bg-emerald-500/20',
        border: 'border-emerald-500/30',
        text: 'text-emerald-400',
        glow: 'shadow-emerald-500/20'
      };
      case 'negative': return {
        bg: 'bg-rose-500/20',
        border: 'border-rose-500/30',
        text: 'text-rose-400',
        glow: 'shadow-rose-500/20'
      };
      default: return {
        bg: 'bg-blue-500/20',
        border: 'border-blue-500/30',
        text: 'text-blue-400',
        glow: 'shadow-blue-500/20'
      };
    }
  };

  const getSentimentIcon = (sentiment?: string) => {
    switch (sentiment) {
      case 'positive': return (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
      case 'negative': return (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
        </svg>
      );
      default: return (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
        </svg>
      );
    }
  };

  const hasNews = newsData && newsData.items && newsData.items.length > 0;

  // Loading state
  if (isLoading) {
    return (
      <div className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <div className="h-10 bg-slate-700/50 rounded-xl w-64 mx-auto mb-4 animate-pulse"></div>
            <div className="h-5 bg-slate-700/50 rounded-lg w-96 mx-auto animate-pulse"></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="rounded-2xl bg-slate-800/50 border border-slate-700/50 p-6">
                <div className="animate-pulse space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-slate-700/50 rounded-xl"></div>
                    <div className="flex-1">
                      <div className="h-4 bg-slate-700/50 rounded w-24 mb-2"></div>
                      <div className="h-3 bg-slate-700/50 rounded w-16"></div>
                    </div>
                  </div>
                  <div className="h-5 bg-slate-700/50 rounded w-full"></div>
                  <div className="h-5 bg-slate-700/50 rounded w-3/4"></div>
                  <div className="h-16 bg-slate-700/50 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-rose-500/20 border border-rose-500/30 flex items-center justify-center">
              <svg className="w-10 h-10 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">Failed to load news</h3>
            <p className="text-slate-400 mb-6 max-w-md mx-auto">Unable to fetch the latest cryptocurrency news. Please try again.</p>
            <button 
              onClick={() => refetch()}
              className="px-6 py-3 rounded-xl font-bold bg-gradient-to-r from-rose-500 to-pink-500 text-white hover:from-rose-600 hover:to-pink-600 transition-all duration-300 hover:scale-105"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative py-20 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900"></div>
      
      {/* Decorative elements */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
      
      {/* Grid pattern */}
      <div className="absolute inset-0 opacity-[0.02]" style={{
        backgroundImage: `linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)`,
        backgroundSize: '40px 40px'
      }}></div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6 mb-12">
          <div>
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
              <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
              <span className="text-cyan-400 text-sm font-bold tracking-wide uppercase">Live Updates</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-black text-white mb-3">
              Market <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">Intelligence</span>
            </h2>
            <p className="text-lg text-slate-400 max-w-xl">
              Real-time crypto news with AI-powered sentiment analysis from trusted sources worldwide.
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            {/* View Mode Toggle */}
            <div className="flex items-center p-1 rounded-xl bg-slate-800/80 border border-slate-700/50">
              <button
                onClick={() => setViewMode('grid')}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 ${
                  viewMode === 'grid' 
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/25' 
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
                Grid
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 ${
                  viewMode === 'list' 
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/25' 
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
                </svg>
                List
              </button>
            </div>
            
            {/* Refresh Button */}
            <button
              onClick={() => refetch()}
              className="group flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm bg-slate-800/80 border border-slate-700/50 text-slate-300 hover:text-white hover:border-cyan-500/50 hover:bg-slate-700/80 transition-all duration-300"
            >
              <svg className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
            
            {hasNews && (
              <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold text-sm">
                <div className="w-2 h-2 rounded-full bg-emerald-400"></div>
                {newsData.items.length} articles
              </div>
            )}
          </div>
        </div>

        {/* News Content */}
        {!hasNews ? (
          <div className="text-center py-20">
            <div className="w-24 h-24 mx-auto mb-8 rounded-3xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-2xl shadow-cyan-500/25">
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
              </svg>
            </div>
            <h3 className="text-3xl font-bold text-white mb-4">Loading Latest News</h3>
            <p className="text-slate-400 max-w-md mx-auto text-lg">
              Fetching the latest cryptocurrency news from trusted sources across the web.
            </p>
          </div>
        ) : (
          <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-6' : 'space-y-4'}>
            {newsData.items.map((article, index) => {
              const articleSource = typeof article.source === 'string' ? article.source : article.source?.name || 'Unknown';
              const sentimentStyles = getSentimentStyles(article.sentiment);
              const isFirst = index === 0;
              
              return viewMode === 'grid' ? (
                /* Grid View Card */
                <article 
                  key={article.id} 
                  className={`group relative rounded-2xl bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 hover:border-slate-600/50 transition-all duration-500 overflow-hidden ${isFirst ? 'md:col-span-2' : ''}`}
                >
                  {/* Hover glow effect */}
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  
                  <div className={`relative p-6 ${isFirst ? 'md:p-8' : ''}`}>
                    {/* Source and sentiment row */}
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                          <span className="text-sm font-black text-white">
                            {articleSource.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <div className="text-sm font-bold text-white">{articleSource}</div>
                          <div className="text-xs text-slate-500">{formatTimeAgo(article.published_at || new Date().toISOString())}</div>
                        </div>
                      </div>
                      
                      {article.sentiment && (
                        <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border ${sentimentStyles.bg} ${sentimentStyles.border} ${sentimentStyles.text}`}>
                          {getSentimentIcon(article.sentiment)}
                          <span>{article.sentiment}</span>
                        </div>
                      )}
                    </div>
                    
                    {/* Title */}
                    <h3 className={`font-bold text-white group-hover:text-cyan-400 transition-colors duration-300 mb-3 leading-tight ${isFirst ? 'text-2xl md:text-3xl' : 'text-lg line-clamp-2'}`}>
                      <a href={article.url} target="_blank" rel="noopener noreferrer" className="hover:underline decoration-cyan-400/50 underline-offset-4">
                        {article.title}
                      </a>
                    </h3>
                    
                    {/* Content preview */}
                    {article.content_text && (
                      <p className={`text-slate-400 leading-relaxed ${isFirst ? 'text-base line-clamp-3' : 'text-sm line-clamp-2'}`}>
                        {article.content_text}
                      </p>
                    )}
                    
                    {/* Read more link */}
                    <div className="mt-4 pt-4 border-t border-slate-700/50">
                      <a 
                        href={article.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 text-sm font-semibold text-cyan-400 hover:text-cyan-300 transition-colors"
                      >
                        Read full article
                        <svg className="w-4 h-4 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                        </svg>
                      </a>
                    </div>
                  </div>
                </article>
              ) : (
                /* List View Card */
                <article 
                  key={article.id} 
                  className="group relative rounded-xl bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 hover:border-slate-600/50 transition-all duration-300 p-5"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-cyan-500/20">
                      <span className="text-sm font-black text-white">
                        {articleSource.charAt(0).toUpperCase()}
                      </span>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-4 mb-2">
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-bold text-white">{articleSource}</span>
                          <span className="text-xs text-slate-500">{formatTimeAgo(article.published_at || new Date().toISOString())}</span>
                        </div>
                        
                        {article.sentiment && (
                          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold border ${sentimentStyles.bg} ${sentimentStyles.border} ${sentimentStyles.text}`}>
                            {getSentimentIcon(article.sentiment)}
                            <span className="hidden sm:inline">{article.sentiment}</span>
                          </div>
                        )}
                      </div>
                      
                      <h3 className="font-bold text-white group-hover:text-cyan-400 transition-colors duration-300 mb-2 line-clamp-2">
                        <a href={article.url} target="_blank" rel="noopener noreferrer" className="hover:underline decoration-cyan-400/50 underline-offset-2">
                          {article.title}
                        </a>
                      </h3>
                      
                      {article.content_text && (
                        <p className="text-sm text-slate-400 line-clamp-2 leading-relaxed">
                          {article.content_text}
                        </p>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
        
        {/* Load More Button */}
        {hasNews && newsData.total > newsData.items.length && (
          <div className="text-center pt-12">
            <button className="group relative px-8 py-4 rounded-xl font-bold text-white overflow-hidden transition-all duration-300 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-blue-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <span className="relative flex items-center gap-2">
                Load More Articles
                <svg className="w-4 h-4 group-hover:translate-y-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default NewsPanel;
