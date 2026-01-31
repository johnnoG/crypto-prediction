import { useState } from 'react';
import Header from './Header';
import HomePage from './pages/HomePage';
import MarketsPage from './pages/MarketsPage';
import ForecastsPage from './pages/ForecastsPage';
import NewsPage from './pages/NewsPage';
import PortfolioPage from './pages/PortfolioPage';
import SettingsPage from './pages/SettingsPage';
import WatchlistPage from './pages/WatchlistPage';
import AlertsPage from './pages/AlertsPage';

function Dashboard() {
  const [currentPage, setCurrentPage] = useState('home');

  const handleNavigate = (page: string) => {
    setCurrentPage(page);
  };

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'markets':
        return <MarketsPage />;
      case 'forecasts':
        return <ForecastsPage />;
      case 'news':
        return <NewsPage />;
      case 'portfolio':
        return <PortfolioPage />;
      case 'settings':
        return <SettingsPage />;
      case 'watchlist':
        return <WatchlistPage />;
      case 'alerts':
        return <AlertsPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <Header currentPage={currentPage} onNavigate={handleNavigate} />
      
      {/* Page Content */}
      {renderCurrentPage()}
    </div>
  );
}

export default Dashboard;