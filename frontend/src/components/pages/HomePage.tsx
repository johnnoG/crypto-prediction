import { useCallback, useState } from 'react';
import PriceTicker from '../PriceTicker';
import RealTimeCryptoGrid from '../RealTimeCryptoGrid';
import ForecastPanel from '../ForecastPanel';
import NewsPanel from '../NewsPanel';
import ScrollToTop from '../ScrollToTop';


function HomePage() {
  const [selectedCrypto, setSelectedCrypto] = useState<string | null>(null);
  const scrollToSection = useCallback((sectionId: string) => {
    const section = document.getElementById(sectionId);
    if (section) {
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, []);

  return (
    <div className="min-h-screen">
      {/* Price Ticker - Sticky below header, hides on scroll down */}
      <PriceTicker />
      
      {/* Hero Section */}
      <section id="hero" className="relative min-h-[90vh] flex items-center overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950"></div>
        
        {/* Animated mesh gradient */}
        <div className="absolute inset-0 opacity-60" style={{
          backgroundImage: `
            radial-gradient(at 20% 30%, rgba(99, 102, 241, 0.3) 0px, transparent 50%),
            radial-gradient(at 80% 20%, rgba(139, 92, 246, 0.25) 0px, transparent 50%),
            radial-gradient(at 40% 80%, rgba(59, 130, 246, 0.2) 0px, transparent 50%),
            radial-gradient(at 90% 70%, rgba(16, 185, 129, 0.15) 0px, transparent 50%)
          `
        }}></div>
        
        {/* Grid pattern */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)`,
          backgroundSize: '60px 60px'
        }}></div>
        
        {/* Glowing orbs */}
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse" style={{animationDuration: '4s'}}></div>
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{animationDuration: '5s', animationDelay: '1s'}}></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-500/10 rounded-full blur-3xl"></div>
        
        {/* Decorative lines */}
        <svg className="absolute inset-0 w-full h-full opacity-10" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="heroLine1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#6366f1" stopOpacity="0" />
              <stop offset="50%" stopColor="#6366f1" stopOpacity="1" />
              <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d="M0,300 Q400,100 800,300 T1600,300" stroke="url(#heroLine1)" strokeWidth="1" fill="none" />
          <path d="M0,400 Q400,200 800,400 T1600,400" stroke="url(#heroLine1)" strokeWidth="1" fill="none" />
        </svg>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            {/* Badge */}
            <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm mb-10 animate-fade-in-down">
              <div className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400"></span>
              </div>
              <span className="text-emerald-400 text-sm font-bold tracking-wider uppercase">Live Platform</span>
              <span className="text-slate-500">•</span>
              <span className="text-slate-400 text-sm">Real-time data streaming</span>
            </div>
            
            {/* Headline */}
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-black mb-8 leading-[0.9] animate-fade-in-down" style={{animationDelay: '0.1s'}}>
              <span className="text-white">The Future of</span>
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400">
                Crypto Intelligence
              </span>
            </h1>
            
            {/* Subtitle */}
            <p className="text-xl md:text-2xl text-slate-400 mb-12 max-w-3xl mx-auto leading-relaxed animate-fade-in-up" style={{animationDelay: '0.2s'}}>
              Professional-grade analytics powered by AI. Real-time market data, 
              predictive forecasts, and actionable intelligence.
            </p>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center animate-fade-in-up" style={{animationDelay: '0.3s'}}>
              <button
                type="button"
                onClick={() => scrollToSection('markets')}
                className="group relative px-8 py-4 rounded-xl font-bold text-lg overflow-hidden transition-all duration-300 hover:scale-105"
                aria-label="Explore live crypto markets"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-600"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-700 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <span className="relative flex items-center justify-center gap-2 text-white">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                  Explore Markets
                </span>
              </button>
              
              <button
                type="button"
                onClick={() => scrollToSection('forecasts')}
                className="group px-8 py-4 rounded-xl font-bold text-lg bg-white/5 border border-white/20 text-white hover:bg-white/10 hover:border-white/30 transition-all duration-300 hover:scale-105"
                aria-label="View AI forecasts"
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  AI Forecasts
                </span>
              </button>
            </div>
            
            {/* Stats */}
            <div className="mt-16 flex flex-wrap justify-center gap-8 md:gap-16 animate-fade-in-up" style={{animationDelay: '0.4s'}}>
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-1">500+</div>
                <div className="text-sm text-slate-500 uppercase tracking-wider">Cryptocurrencies</div>
              </div>
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-1">&lt;1s</div>
                <div className="text-sm text-slate-500 uppercase tracking-wider">Data Latency</div>
              </div>
              <div className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-white mb-1">24/7</div>
                <div className="text-sm text-slate-500 uppercase tracking-wider">Market Coverage</div>
              </div>
            </div>
            
            {/* Scroll Indicator */}
            <div className="mt-16 animate-bounce">
              <button 
                type="button"
                onClick={() => scrollToSection('markets')}
                className="p-3 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 transition-colors"
                aria-label="Scroll to markets"
              >
                <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content - Continuous Scroll */}
      <main className="relative">
        {/* Markets Section */}
        <section id="markets" className="py-24 relative overflow-hidden">
          {/* Background */}
          <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900"></div>
          
          {/* Animated gradient orbs */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-emerald-500/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-blue-500/10 rounded-full blur-3xl"></div>
          
          {/* Grid pattern */}
          <div className="absolute inset-0 opacity-[0.02]" style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)`,
            backgroundSize: '50px 50px'
          }}></div>
          
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Header */}
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 backdrop-blur-sm mb-8">
                <div className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400"></span>
                </div>
                <span className="text-emerald-400 text-sm font-bold tracking-widest uppercase">Live Market Data</span>
              </div>
              
              <h2 className="text-5xl md:text-7xl font-black mb-6 leading-tight">
                <span className="text-white">Crypto </span>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400">Markets</span>
              </h2>
              
              <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10">
                Real-time prices, market caps, and trading volumes for the top cryptocurrencies.
              </p>
              
              {/* Stats row */}
              <div className="flex flex-wrap justify-center gap-6 md:gap-12">
                <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10">
                  <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                    <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <div className="text-left">
                    <div className="text-2xl font-bold text-white">$2.3T</div>
                    <div className="text-xs text-slate-500">Total Market Cap</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                    <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <div className="text-left">
                    <div className="text-2xl font-bold text-white">$89B</div>
                    <div className="text-xs text-slate-500">24h Volume</div>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10">
                  <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                    <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                  </div>
                  <div className="text-left">
                    <div className="text-2xl font-bold text-white">500+</div>
                    <div className="text-xs text-slate-500">Cryptocurrencies</div>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Crypto Grid with glow effect */}
            <div className="relative">
              <div className="absolute -inset-4 bg-gradient-to-r from-emerald-500/10 via-teal-500/10 to-cyan-500/10 rounded-3xl blur-2xl opacity-50"></div>
              <div className="relative">
                <RealTimeCryptoGrid 
                  selectedCrypto={selectedCrypto}
                  onSelectCrypto={setSelectedCrypto}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Forecasts Section */}
        <section id="forecasts" className="py-24 relative overflow-hidden">
          {/* Background */}
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-950 via-purple-900 to-slate-900"></div>
          
          {/* Animated mesh gradient */}
          <div className="absolute inset-0 opacity-40" style={{
            backgroundImage: `
              radial-gradient(at 10% 20%, rgba(99, 102, 241, 0.4) 0px, transparent 50%),
              radial-gradient(at 90% 80%, rgba(168, 85, 247, 0.3) 0px, transparent 50%),
              radial-gradient(at 50% 50%, rgba(79, 70, 229, 0.2) 0px, transparent 50%)
            `
          }}></div>
          
          {/* Floating elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-20 left-[10%] w-64 h-64 bg-indigo-500/20 rounded-full blur-3xl animate-pulse"></div>
            <div className="absolute bottom-20 right-[10%] w-80 h-80 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            
            {/* Floating chart lines decoration */}
            <svg className="absolute top-1/4 left-0 w-full h-32 opacity-10" viewBox="0 0 1200 100" preserveAspectRatio="none">
              <path d="M0,50 Q300,20 600,60 T1200,40" stroke="url(#gradient1)" strokeWidth="2" fill="none" />
              <path d="M0,70 Q400,30 800,70 T1200,50" stroke="url(#gradient2)" strokeWidth="2" fill="none" />
              <defs>
                <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#818cf8" />
                  <stop offset="100%" stopColor="#c084fc" />
                </linearGradient>
                <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#a855f7" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          
          {/* Grid overlay */}
          <div className="absolute inset-0 opacity-[0.03]" style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)`,
            backgroundSize: '60px 60px'
          }}></div>
          
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Header */}
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 backdrop-blur-sm mb-8">
                <div className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-400"></span>
                </div>
                <span className="text-indigo-300 text-sm font-bold tracking-widest uppercase">AI-Powered Predictions</span>
              </div>
              
              <h2 className="text-5xl md:text-7xl font-black mb-6 leading-tight">
                <span className="text-white">Price </span>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">Forecasts</span>
              </h2>
              
              <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10">
                Advanced machine learning models analyze market patterns to predict future price movements with confidence intervals.
              </p>
              
              {/* Feature pills */}
              <div className="flex flex-wrap justify-center gap-3">
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-slate-300">
                  <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  <span>ARIMA Models</span>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-slate-300">
                  <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <span>Technical Analysis</span>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-sm text-slate-300">
                  <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Real-time Updates</span>
                </div>
              </div>
            </div>
            
            {/* Forecast Panel */}
            <div className="relative">
              {/* Glow effect behind panel */}
              <div className="absolute -inset-4 bg-gradient-to-r from-indigo-500/20 via-purple-500/20 to-pink-500/20 rounded-3xl blur-2xl opacity-50"></div>
              <div className="relative">
                <ForecastPanel />
              </div>
            </div>
          </div>
        </section>

        {/* News Section */}
        <section id="news">
          <NewsPanel />
        </section>

        {/* Features Section - Why Choose Us */}
        <section id="features" className="py-24 relative overflow-hidden">
          {/* Animated Background */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-800 to-slate-900"></div>
          <div className="absolute inset-0 opacity-30" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%234f46e5' fill-opacity='0.15' fill-rule='evenodd'/%3E%3C/svg%3E")`
          }}></div>
          
          {/* Glowing orbs */}
          <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-500/10 rounded-full blur-3xl"></div>
          
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Header */}
            <div className="text-center mb-20">
              <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 backdrop-blur-sm mb-8">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
                <span className="text-indigo-300 text-sm font-bold tracking-widest uppercase">Why Choose Us</span>
              </div>
              <h2 className="text-5xl md:text-7xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white via-indigo-200 to-purple-200 mb-6 leading-tight">
                Built for Traders,<br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">By Traders</span>
              </h2>
              <p className="text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed">
                Professional-grade tools that give you the edge in crypto markets
              </p>
            </div>
            
            {/* Stats Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-20">
              {[
                { value: '500+', label: 'Cryptocurrencies', color: 'from-blue-400 to-cyan-400' },
                { value: '<1s', label: 'Data Latency', color: 'from-emerald-400 to-teal-400' },
                { value: '87%', label: 'Forecast Accuracy', color: 'from-purple-400 to-pink-400' },
                { value: '24/7', label: 'Market Coverage', color: 'from-orange-400 to-amber-400' },
              ].map((stat, index) => (
                <div 
                  key={index}
                  className="group relative p-6 rounded-2xl bg-white/5 backdrop-blur-sm border border-white/10 hover:border-white/20 transition-all duration-500 hover:scale-105"
                >
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                  <div className={`text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r ${stat.color} mb-2`}>
                    {stat.value}
                  </div>
                  <div className="text-slate-400 text-sm font-medium">{stat.label}</div>
                </div>
              ))}
            </div>
            
            {/* Feature Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Card 1 - Real-time Data */}
              <div className="group relative">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-3xl opacity-0 group-hover:opacity-100 blur transition-all duration-500"></div>
                <div className="relative h-full p-8 rounded-3xl bg-slate-800/80 backdrop-blur-xl border border-slate-700/50 hover:border-transparent transition-all duration-500">
                  {/* Icon */}
                  <div className="w-16 h-16 mb-8 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/25 group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  
                  {/* Content */}
                  <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-blue-400 group-hover:to-cyan-400 transition-all duration-300">
                    Real-time Data
                  </h3>
                  <p className="text-slate-400 leading-relaxed mb-6">
                    Live cryptocurrency prices with sub-second updates. Market cap, volume, and price changes streamed directly to your dashboard.
                  </p>
                  
                  {/* Features List */}
                  <ul className="space-y-3">
                    {['WebSocket streaming', 'Smart caching', 'Offline fallback'].map((feature, i) => (
                      <li key={i} className="flex items-center text-sm text-slate-300">
                        <svg className="w-4 h-4 mr-3 text-cyan-400" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Card 2 - AI Forecasting */}
              <div className="group relative lg:-translate-y-4">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-3xl opacity-0 group-hover:opacity-100 blur transition-all duration-500"></div>
                <div className="relative h-full p-8 rounded-3xl bg-slate-800/80 backdrop-blur-xl border border-slate-700/50 hover:border-transparent transition-all duration-500">
                  {/* Popular Badge */}
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <div className="px-4 py-1 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs font-bold uppercase tracking-wider shadow-lg">
                      Most Popular
                    </div>
                  </div>
                  
                  {/* Icon */}
                  <div className="w-16 h-16 mb-8 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/25 group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                  
                  {/* Content */}
                  <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-purple-400 group-hover:to-pink-400 transition-all duration-300">
                    AI Forecasting
                  </h3>
                  <p className="text-slate-400 leading-relaxed mb-6">
                    Advanced machine learning models analyze patterns and predict price movements with confidence intervals.
                  </p>
                  
                  {/* Features List */}
                  <ul className="space-y-3">
                    {['ARIMA & Prophet models', 'Confidence bands', 'Backtesting metrics'].map((feature, i) => (
                      <li key={i} className="flex items-center text-sm text-slate-300">
                        <svg className="w-4 h-4 mr-3 text-pink-400" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Card 3 - Market Intelligence */}
              <div className="group relative">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-3xl opacity-0 group-hover:opacity-100 blur transition-all duration-500"></div>
                <div className="relative h-full p-8 rounded-3xl bg-slate-800/80 backdrop-blur-xl border border-slate-700/50 hover:border-transparent transition-all duration-500">
                  {/* Icon */}
                  <div className="w-16 h-16 mb-8 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/25 group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                    </svg>
                  </div>
                  
                  {/* Content */}
                  <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-emerald-400 group-hover:to-teal-400 transition-all duration-300">
                    Market Intelligence
                  </h3>
                  <p className="text-slate-400 leading-relaxed mb-6">
                    Curated news aggregation with sentiment analysis. Stay ahead with real-time market insights and trends.
                  </p>
                  
                  {/* Features List */}
                  <ul className="space-y-3">
                    {['News aggregation', 'Sentiment scoring', 'Topic classification'].map((feature, i) => (
                      <li key={i} className="flex items-center text-sm text-slate-300">
                        <svg className="w-4 h-4 mr-3 text-teal-400" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-32 relative overflow-hidden">
          {/* Animated gradient background */}
          <div className="absolute inset-0 bg-gradient-to-br from-violet-950 via-purple-900 to-fuchsia-950"></div>
          
          {/* Animated mesh gradient overlay */}
          <div className="absolute inset-0 opacity-60" style={{
            backgroundImage: `
              radial-gradient(at 20% 30%, rgba(139, 92, 246, 0.4) 0px, transparent 50%),
              radial-gradient(at 80% 20%, rgba(236, 72, 153, 0.3) 0px, transparent 50%),
              radial-gradient(at 40% 80%, rgba(59, 130, 246, 0.3) 0px, transparent 50%),
              radial-gradient(at 90% 70%, rgba(168, 85, 247, 0.4) 0px, transparent 50%)
            `
          }}></div>
          
          {/* Floating geometric shapes */}
          <div className="absolute inset-0 overflow-hidden">
            {/* Animated rings */}
            <div className="absolute top-1/4 left-1/4 w-96 h-96 border border-white/10 rounded-full animate-pulse" style={{ animationDuration: '4s' }}></div>
            <div className="absolute top-1/4 left-1/4 w-80 h-80 border border-white/5 rounded-full translate-x-8 translate-y-8"></div>
            <div className="absolute bottom-1/4 right-1/4 w-72 h-72 border border-white/10 rounded-full animate-pulse" style={{ animationDuration: '3s' }}></div>
            
            {/* Glowing orbs */}
            <div className="absolute top-10 right-20 w-64 h-64 bg-pink-500/20 rounded-full blur-3xl animate-pulse"></div>
            <div className="absolute bottom-10 left-20 w-80 h-80 bg-violet-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-blue-500/10 rounded-full blur-3xl"></div>
            
            {/* Floating particles */}
            <div className="absolute top-20 left-[10%] w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDuration: '3s' }}></div>
            <div className="absolute top-40 right-[20%] w-3 h-3 bg-pink-400/50 rounded-full animate-bounce" style={{ animationDuration: '2.5s', animationDelay: '0.5s' }}></div>
            <div className="absolute bottom-32 left-[30%] w-2 h-2 bg-violet-400/50 rounded-full animate-bounce" style={{ animationDuration: '2s', animationDelay: '1s' }}></div>
            <div className="absolute top-1/3 right-[15%] w-1.5 h-1.5 bg-white/30 rounded-full animate-bounce" style={{ animationDuration: '3.5s' }}></div>
            <div className="absolute bottom-20 right-[35%] w-2 h-2 bg-fuchsia-400/40 rounded-full animate-bounce" style={{ animationDuration: '2.8s', animationDelay: '0.3s' }}></div>
          </div>
          
          {/* Grid pattern overlay */}
          <div className="absolute inset-0 opacity-[0.03]" style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)`,
            backgroundSize: '50px 50px'
          }}></div>
          
          <div className="relative max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/10 border border-white/20 backdrop-blur-sm mb-10">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400"></span>
              </span>
              <span className="text-white/90 text-sm font-semibold tracking-wide">Live Now — Real-time market data</span>
            </div>
            
            {/* Main heading with gradient */}
            <h2 className="text-5xl md:text-7xl font-black mb-8 leading-tight">
              <span className="text-white">Ready to </span>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 via-purple-400 to-indigo-400 animate-pulse" style={{ animationDuration: '3s' }}>
                dominate
              </span>
              <br />
              <span className="text-white">the market?</span>
            </h2>
            
            <p className="text-xl md:text-2xl text-white/70 mb-12 max-w-2xl mx-auto leading-relaxed">
              Join the next generation of traders using <span className="text-white font-semibold">AI-powered analytics</span> to stay ahead of the curve.
            </p>
            
            {/* Stats mini row */}
            <div className="flex justify-center gap-8 md:gap-16 mb-12">
              {[
                { value: '10K+', label: 'Active Traders' },
                { value: '$2B+', label: 'Volume Tracked' },
                { value: '99.9%', label: 'Uptime' },
              ].map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-white">{stat.value}</div>
                  <div className="text-sm text-white/50">{stat.label}</div>
                </div>
              ))}
            </div>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button 
                type="button"
                onClick={() => scrollToSection('markets')}
                className="group relative px-10 py-5 rounded-2xl font-bold text-lg overflow-hidden transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-pink-500/25"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-pink-600 via-purple-600 to-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <span className="relative text-white flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                  Explore Markets
                </span>
              </button>
              
              <button 
                type="button"
                onClick={() => scrollToSection('forecasts')}
                className="group px-10 py-5 rounded-2xl font-bold text-lg bg-white/10 border border-white/20 backdrop-blur-sm text-white hover:bg-white/20 hover:border-white/30 transition-all duration-300 hover:scale-105"
              >
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  View AI Forecasts
                </span>
              </button>
            </div>
            
            {/* Trust badges */}
            <div className="mt-16 flex flex-wrap justify-center items-center gap-8 opacity-60">
              <span className="text-white/50 text-sm">Trusted by traders worldwide</span>
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-1">
                  {[...Array(5)].map((_, i) => (
                    <svg key={i} className="w-4 h-4 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                  ))}
                </div>
                <span className="text-white/70 text-sm font-medium">4.9/5 from 2,000+ reviews</span>
              </div>
            </div>
          </div>
        </section>
      </main>
      
      {/* Scroll to Top Button */}
      <ScrollToTop />
    </div>
  );
}

export default HomePage;
