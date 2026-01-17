import { useState } from 'react';
import { useCryptoPrices } from '../../hooks/useCryptoPrices';
import CryptoIcon from '../CryptoIcon';

interface PortfolioItem {
  id: string;
  symbol: string;
  name: string;
  amount: number;
  avgBuyPrice: number;
}

function PortfolioPage() {
  const [portfolioItems] = useState<PortfolioItem[]>([
    { id: 'bitcoin', symbol: 'BTC', name: 'Bitcoin', amount: 0.5, avgBuyPrice: 35000 },
    { id: 'ethereum', symbol: 'ETH', name: 'Ethereum', amount: 2.5, avgBuyPrice: 2200 },
    { id: 'solana', symbol: 'SOL', name: 'Solana', amount: 10, avgBuyPrice: 80 },
    { id: 'cardano', symbol: 'ADA', name: 'Cardano', amount: 1000, avgBuyPrice: 0.45 },
  ]);

  const { data: pricesData } = useCryptoPrices(['bitcoin', 'ethereum', 'solana', 'cardano']);

  const calculatePortfolioValue = () => {
    if (!pricesData) return { totalValue: 0, totalGainLoss: 0, totalGainLossPercent: 0 };

    let totalValue = 0;
    let totalCost = 0;

    portfolioItems.forEach(item => {
      const currentPrice = pricesData[item.id]?.usd || 0;
      const itemValue = item.amount * currentPrice;
      const itemCost = item.amount * item.avgBuyPrice;
      
      totalValue += itemValue;
      totalCost += itemCost;
    });

    const totalGainLoss = totalValue - totalCost;
    const totalGainLossPercent = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0;

    return { totalValue, totalGainLoss, totalGainLossPercent };
  };

  const { totalValue, totalGainLoss, totalGainLossPercent } = calculatePortfolioValue();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50">
      {/* Page Header */}
      <div className="bg-gradient-to-br from-purple-600 via-pink-700 to-red-800 py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-red-600/20"></div>
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'60\' height=\'60\' viewBox=\'0 0 60 60\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' fill-rule=\'evenodd\'%3E%3Cg fill=\'%23ffffff\' fill-opacity=\'0.05\'%3E%3Ccircle cx=\'30\' cy=\'30\' r=\'2\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'
        }}></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-white/10 backdrop-blur-md border border-white/20 mb-6">
              <span className="text-white text-sm font-semibold flex items-center">
                <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M4 4a2 2 0 00-2 2v1h16V6a2 2 0 00-2-2H4z" />
                  <path fillRule="evenodd" d="M18 9H2v5a2 2 0 002 2h12a2 2 0 002-2V9zM4 13a1 1 0 011-1h1a1 1 0 110 2H5a1 1 0 01-1-1zm5-1a1 1 0 100 2h1a1 1 0 100-2H9z" clipRule="evenodd" />
                </svg>
                Investment Tracking
              </span>
            </div>
            <h1 className="text-5xl md:text-6xl font-black text-white mb-6 leading-tight">
              Your Portfolio
            </h1>
            <p className="text-xl text-purple-100 max-w-3xl mx-auto leading-relaxed">
              Track your cryptocurrency investments and monitor performance in real-time with advanced analytics.
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Portfolio Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Total Portfolio Value</h3>
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
            <p className="text-4xl font-black text-gray-900">{formatCurrency(totalValue)}</p>
          </div>
          
          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Total Gain/Loss</h3>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                totalGainLoss >= 0 ? 'bg-gradient-to-br from-green-500 to-green-600' : 'bg-gradient-to-br from-red-500 to-red-600'
              }`}>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={totalGainLoss >= 0 ? "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" : "M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6"} />
                </svg>
              </div>
            </div>
            <p className={`text-4xl font-black ${totalGainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatCurrency(totalGainLoss)}
            </p>
          </div>
          
          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Total Return</h3>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                totalGainLossPercent >= 0 ? 'bg-gradient-to-br from-purple-500 to-purple-600' : 'bg-gradient-to-br from-orange-500 to-orange-600'
              }`}>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
            <p className={`text-4xl font-black ${totalGainLossPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatPercentage(totalGainLossPercent)}
            </p>
          </div>
        </div>

      {/* Holdings */}
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
        <div className="px-8 py-6 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-gray-100">
          <h3 className="text-2xl font-black text-gray-900">Your Holdings</h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Asset</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Current Price</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg Buy Price</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Market Value</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Gain/Loss</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Return</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {portfolioItems.map((item) => {
                const currentPrice = pricesData?.[item.id]?.usd || 0;
                const marketValue = item.amount * currentPrice;
                const costBasis = item.amount * item.avgBuyPrice;
                const gainLoss = marketValue - costBasis;
                const gainLossPercent = costBasis > 0 ? (gainLoss / costBasis) * 100 : 0;

                return (
                  <tr key={item.id} className="hover:bg-gradient-to-r hover:from-purple-50 hover:via-pink-50 hover:to-purple-50 transition-all duration-300">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-3">
                        <CryptoIcon cryptoId={item.id} size="md" />
                        <div>
                          <div className="text-lg font-black text-gray-900">{item.name}</div>
                          <div className="text-sm font-bold text-gray-500">{item.symbol}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-base font-black text-gray-900">
                      {item.amount.toLocaleString()} {item.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-base font-black text-gray-900">
                      {formatCurrency(currentPrice)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-base font-black text-gray-500">
                      {formatCurrency(item.avgBuyPrice)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-lg font-black text-gray-900">
                      {formatCurrency(marketValue)}
                    </td>
                    <td className={`px-6 py-4 whitespace-nowrap text-base font-black ${gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(gainLoss)}
                    </td>
                    <td className={`px-6 py-4 whitespace-nowrap text-base font-black ${gainLossPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercentage(gainLossPercent)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Coming Soon Notice */}
      <div className="mt-12 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-10 text-center border border-blue-100 shadow-lg">
        <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-xl">
          <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
        </div>
        <h3 className="text-2xl font-black text-blue-900 mb-4">Portfolio Features Coming Soon</h3>
        <p className="text-lg text-blue-700 max-w-2xl mx-auto mb-6">
          Add/remove holdings, transaction history, performance analytics, and automated portfolio rebalancing recommendations.
        </p>
        <button className="coinbase-button">
          Join Waitlist
        </button>
      </div>
      </div>
    </div>
  );
}

export default PortfolioPage;
