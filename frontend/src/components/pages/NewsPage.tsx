import NewsPanel from '../NewsPanel';

function NewsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-indigo-50 dark:from-gray-900 dark:via-gray-900 dark:to-gray-900">
      {/* Page Header */}
      <div className="bg-gradient-to-br from-indigo-600 via-purple-700 to-pink-800 dark:from-indigo-800 dark:via-purple-800 dark:to-pink-900 py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/20 to-pink-600/20"></div>
        <div className="absolute inset-0 opacity-20" style={{
          backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'60\' height=\'60\' viewBox=\'0 0 60 60\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' fill-rule=\'evenodd\'%3E%3Cg fill=\'%23ffffff\' fill-opacity=\'0.05\'%3E%3Ccircle cx=\'30\' cy=\'30\' r=\'2\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")'
        }}></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-white/10 backdrop-blur-md border border-white/20 mb-6">
              <span className="text-white text-sm font-semibold flex items-center">
                <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M2 5a2 2 0 012-2h8a2 2 0 012 2v10a2 2 0 002 2H4a2 2 0 01-2-2V5zm3 1h6v4H5V6zm6 6H5v2h6v-2z" clipRule="evenodd" />
                  <path d="M15 7h1a2 2 0 012 2v5.5a1.5 1.5 0 01-3 0V7z" />
                </svg>
                Latest Market Intelligence
              </span>
            </div>
            <h1 className="text-5xl md:text-6xl font-black text-white mb-6 leading-tight">
              Crypto News & Updates
            </h1>
            <p className="text-xl text-indigo-100 max-w-3xl mx-auto leading-relaxed">
              Stay informed with real-time cryptocurrency news, market analysis, and blockchain technology updates from trusted sources.
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Stats Bar */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700 text-center">
            <div className="text-3xl font-black text-gray-900 dark:text-gray-100 mb-2">100+</div>
            <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">Articles Today</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700 text-center">
            <div className="text-3xl font-black text-green-600 dark:text-green-400 mb-2">85%</div>
            <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">Positive Sentiment</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700 text-center">
            <div className="text-3xl font-black text-blue-600 dark:text-blue-400 mb-2">15</div>
            <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">News Sources</div>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-lg border border-gray-100 dark:border-gray-700 text-center">
            <div className="text-3xl font-black text-purple-600 dark:text-purple-400 mb-2">5 min</div>
            <div className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">Update Interval</div>
          </div>
        </div>

        {/* News Panel */}
        <NewsPanel />
      </div>
    </div>
  );
}

export default NewsPage;
