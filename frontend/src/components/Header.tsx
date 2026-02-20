import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import SignInForm from './auth/SignInForm';
import SignUpForm from './auth/SignUpForm';
import UserMenu from './auth/UserMenu';

interface HeaderProps {
  currentPage?: string;
  onNavigate?: (page: string) => void;
}

function Header({ currentPage, onNavigate }: HeaderProps) {
  const { isAuthenticated } = useAuth();
  const [showSignIn, setShowSignIn] = useState(false);
  const [showSignUp, setShowSignUp] = useState(false);

  const PROTECTED_PAGES = ['markets', 'forecasts', 'news', 'watchlist'];

  // Handle navigation - use onNavigate if provided, otherwise scroll to section
  const handleNavigation = (pageId: string) => {
    if (PROTECTED_PAGES.includes(pageId) && !isAuthenticated) {
      setShowSignIn(true);
      return;
    }
    if (onNavigate) {
      onNavigate(pageId);
    } else {
      const element = document.getElementById(pageId);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  const navItems = [
    {
      id: 'home',
      label: 'Home',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5v4m8-4v4" />
        </svg>
      )
    },
    {
      id: 'markets',
      label: 'Markets',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      )
    },
    { 
      id: 'forecasts', 
      label: 'Forecasts', 
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      )
    },
    { 
      id: 'news', 
      label: 'News', 
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
        </svg>
      )
    },
    {
      id: 'watchlist',
      label: 'Watchlist',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
        </svg>
      )
    },
  ];

  return (
    <header className="sticky top-0 z-50">
      {/* Glass effect background */}
      <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-xl"></div>
      
      {/* Gradient border bottom */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent"></div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center gap-4">
            <button
              onClick={() => handleNavigation('home')}
              className="group relative"
            >
              {/* Logo glow effect */}
              <div className="absolute -inset-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl opacity-0 group-hover:opacity-30 blur-lg transition-all duration-500"></div>
              <div className="relative w-10 h-10 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg transform group-hover:scale-105 transition-all duration-300">
                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 32 32">
                  <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm7.189-17.98c.314-2.096-1.283-3.223-3.465-3.975l.708-2.84-1.728-.43-.69 2.765c-.454-.114-.92-.22-1.385-.326l.695-2.783L15.596 6l-.708 2.839c-.376-.086-.746-.17-1.104-.26l.002-.009-2.384-.595-.46 1.846s1.283.294 1.256.312c.7.175.826.638.805 1.006l-.806 3.235c.048.012.11.03.18.057l-.183-.045-1.13 4.532c-.086.212-.303.531-.793.41.018.025-1.256-.313-1.256-.313l-.858 1.978 2.25.561c.418.105.828.215 1.231.318l-.715 2.872 1.727.43.708-2.84c.472.127.93.245 1.378.357l-.706 2.828 1.728.43.715-2.866c2.948.558 5.164.333 6.097-2.333.752-2.146-.037-3.385-1.588-4.192 1.13-.26 1.98-1.003 2.207-2.538zm-3.95 5.538c-.533 2.147-4.148.986-5.32.695l.95-3.805c1.172.293 4.929.872 4.37 3.11zm.535-5.569c-.487 1.953-3.495.96-4.47.717l.86-3.45c.975.243 4.118.696 3.61 2.733z" />
                </svg>
              </div>
            </button>
            
            <div className="hidden sm:block">
              <button
                onClick={() => handleNavigation('home')}
                className="text-xl font-bold text-white hover:text-blue-400 transition-colors"
              >
                CryptoForecast
              </button>
              <p className="text-xs text-slate-500 font-medium tracking-wider uppercase">Pro Trading</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center">
            <div className="flex items-center bg-white/5 rounded-xl p-1 border border-white/10">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => handleNavigation(item.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    currentPage === item.id
                      ? 'bg-blue-500/20 text-blue-400 shadow-lg shadow-blue-500/10'
                      : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {!isAuthenticated && PROTECTED_PAGES.includes(item.id) && (
                    <svg className="w-3 h-3 text-slate-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          </nav>

          {/* Right Section */}
          <div className="flex items-center gap-4">
            {/* Search */}
            <div className="hidden lg:block">
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl opacity-0 group-hover:opacity-30 blur transition-all duration-300"></div>
                <div className="relative flex items-center">
                  <svg className="absolute left-3 w-4 h-4 text-slate-500 group-hover:text-blue-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <input 
                    type="text" 
                    placeholder="Search..."
                    className="w-48 pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-xl text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-blue-500/50 focus:bg-white/10 transition-all duration-300"
                  />
                </div>
              </div>
            </div>
            
            {/* Auth Buttons */}
            <div className="flex items-center gap-2">
              {isAuthenticated ? (
                <UserMenu onNavigate={onNavigate} />
              ) : (
                <>
                  <button 
                    onClick={() => setShowSignIn(true)}
                    className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white transition-colors"
                    aria-label="Sign in to your account"
                  >
                    Sign In
                  </button>
                  <button 
                    onClick={() => setShowSignUp(true)}
                    className="group relative px-4 py-2 rounded-lg font-medium text-sm overflow-hidden"
                    aria-label="Create a new account"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-600"></div>
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-700 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    <span className="relative text-white">Get Started</span>
                  </button>
                </>
              )}
            </div>
            
            {/* Mobile menu button */}
            <button className="md:hidden p-2 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-all">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      {/* Authentication Modals */}
      {showSignIn && (
        <SignInForm 
          onClose={() => setShowSignIn(false)}
          onSwitchToSignUp={() => {
            setShowSignIn(false);
            setShowSignUp(true);
          }}
        />
      )}
      
      {showSignUp && (
        <SignUpForm 
          onClose={() => setShowSignUp(false)}
          onSwitchToSignIn={() => {
            setShowSignUp(false);
            setShowSignIn(true);
          }}
        />
      )}
    </header>
  );
}

export default Header;
