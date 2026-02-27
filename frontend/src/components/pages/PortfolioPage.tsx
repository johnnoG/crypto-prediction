import { useState } from 'react';
import { useCryptoPrices } from '../../hooks/useCryptoPrices';
import { usePortfolio, useAddHolding, useUpdateHolding, useDeleteHolding, PortfolioHolding } from '../../hooks/usePortfolio';
import CryptoIcon from '../CryptoIcon';

// Known CoinGecko IDs for the dropdown
const KNOWN_COINS = [
  { id: 'bitcoin', symbol: 'BTC', name: 'Bitcoin' },
  { id: 'ethereum', symbol: 'ETH', name: 'Ethereum' },
  { id: 'binancecoin', symbol: 'BNB', name: 'BNB' },
  { id: 'solana', symbol: 'SOL', name: 'Solana' },
  { id: 'ripple', symbol: 'XRP', name: 'XRP' },
  { id: 'cardano', symbol: 'ADA', name: 'Cardano' },
  { id: 'dogecoin', symbol: 'DOGE', name: 'Dogecoin' },
  { id: 'polkadot', symbol: 'DOT', name: 'Polkadot' },
  { id: 'chainlink', symbol: 'LINK', name: 'Chainlink' },
  { id: 'litecoin', symbol: 'LTC', name: 'Litecoin' },
  { id: 'avalanche-2', symbol: 'AVAX', name: 'Avalanche' },
  { id: 'matic-network', symbol: 'MATIC', name: 'Polygon' },
  { id: 'uniswap', symbol: 'UNI', name: 'Uniswap' },
  { id: 'cosmos', symbol: 'ATOM', name: 'Cosmos' },
  { id: 'near', symbol: 'NEAR', name: 'NEAR Protocol' },
];

interface HoldingFormState {
  crypto_id: string;
  crypto_symbol: string;
  crypto_name: string;
  amount: string;
  avg_buy_price: string;
}

const EMPTY_FORM: HoldingFormState = {
  crypto_id: '',
  crypto_symbol: '',
  crypto_name: '',
  amount: '',
  avg_buy_price: '',
};

function PortfolioPage() {
  const { data: holdings = [], isLoading, isError } = usePortfolio();
  const addHolding = useAddHolding();
  const updateHolding = useUpdateHolding();
  const deleteHolding = useDeleteHolding();

  const [showAddForm, setShowAddForm] = useState(false);
  const [form, setForm] = useState<HoldingFormState>(EMPTY_FORM);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<{ amount: string; avg_buy_price: string }>({ amount: '', avg_buy_price: '' });
  const [formError, setFormError] = useState('');

  // Fetch live prices for all held coins
  const coinGeckoIds = holdings.map((h: PortfolioHolding) => h.crypto_id);
  const { data: pricesData } = useCryptoPrices(coinGeckoIds.length > 0 ? coinGeckoIds : ['bitcoin']);

  // Summary calculations
  const summary = holdings.reduce(
    (acc: { totalValue: number; totalCost: number }, h: PortfolioHolding) => {
      const currentPrice = (pricesData as Record<string, { usd: number }>)?.[h.crypto_id]?.usd ?? 0;
      acc.totalValue += h.amount * currentPrice;
      acc.totalCost += h.amount * h.avg_buy_price;
      return acc;
    },
    { totalValue: 0, totalCost: 0 },
  );
  const totalGainLoss = summary.totalValue - summary.totalCost;
  const totalGainLossPercent = summary.totalCost > 0 ? (totalGainLoss / summary.totalCost) * 100 : 0;

  const fmt = (v: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }).format(v);
  const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

  const handleCoinSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const coin = KNOWN_COINS.find(c => c.id === e.target.value);
    if (coin) setForm(f => ({ ...f, crypto_id: coin.id, crypto_symbol: coin.symbol, crypto_name: coin.name }));
  };

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError('');
    const amount = parseFloat(form.amount);
    const price = parseFloat(form.avg_buy_price);
    if (!form.crypto_id || isNaN(amount) || amount <= 0 || isNaN(price) || price <= 0) {
      setFormError('Please fill in all fields with valid positive numbers.');
      return;
    }
    try {
      await addHolding.mutateAsync({
        crypto_id: form.crypto_id,
        crypto_symbol: form.crypto_symbol,
        crypto_name: form.crypto_name,
        amount,
        avg_buy_price: price,
      });
      setForm(EMPTY_FORM);
      setShowAddForm(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setFormError(msg.includes('409') ? 'That coin is already in your portfolio. Edit it below.' : 'Failed to add holding.');
    }
  };

  const startEdit = (h: PortfolioHolding) => {
    setEditingId(h.id);
    setEditForm({ amount: String(h.amount), avg_buy_price: String(h.avg_buy_price) });
  };

  const handleUpdate = async (id: number) => {
    const amount = parseFloat(editForm.amount);
    const price = parseFloat(editForm.avg_buy_price);
    if (isNaN(amount) || amount <= 0 || isNaN(price) || price <= 0) return;
    await updateHolding.mutateAsync({ id, data: { amount, avg_buy_price: price } });
    setEditingId(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50">
      {/* Page Header */}
      <div className="bg-gradient-to-br from-purple-600 via-pink-700 to-red-800 py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-red-600/20" />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-white/10 backdrop-blur-md border border-white/20 mb-6">
            <span className="text-white text-sm font-semibold flex items-center">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path d="M4 4a2 2 0 00-2 2v1h16V6a2 2 0 00-2-2H4z" />
                <path fillRule="evenodd" d="M18 9H2v5a2 2 0 002 2h12a2 2 0 002-2V9zM4 13a1 1 0 011-1h1a1 1 0 110 2H5a1 1 0 01-1-1zm5-1a1 1 0 100 2h1a1 1 0 100-2H9z" clipRule="evenodd" />
              </svg>
              Investment Tracking
            </span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black text-white mb-6 leading-tight">Your Portfolio</h1>
          <p className="text-xl text-purple-100 max-w-3xl mx-auto leading-relaxed">
            Track your cryptocurrency investments and monitor performance in real-time.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Summary Cards */}
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
            <p className="text-4xl font-black text-gray-900">{fmt(summary.totalValue)}</p>
          </div>

          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Total Gain/Loss</h3>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${totalGainLoss >= 0 ? 'bg-gradient-to-br from-green-500 to-green-600' : 'bg-gradient-to-br from-red-500 to-red-600'}`}>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={totalGainLoss >= 0 ? "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" : "M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6"} />
                </svg>
              </div>
            </div>
            <p className={`text-4xl font-black ${totalGainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {fmt(totalGainLoss)}
            </p>
          </div>

          <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Total Return</h3>
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${totalGainLossPercent >= 0 ? 'bg-gradient-to-br from-purple-500 to-purple-600' : 'bg-gradient-to-br from-orange-500 to-orange-600'}`}>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
            <p className={`text-4xl font-black ${totalGainLossPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {fmtPct(totalGainLossPercent)}
            </p>
          </div>
        </div>

        {/* Holdings Table */}
        <div className="bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-100">
          <div className="px-8 py-6 border-b border-gray-200 bg-gradient-to-r from-gray-50 to-gray-100 flex items-center justify-between">
            <h3 className="text-2xl font-black text-gray-900">Your Holdings</h3>
            <button
              onClick={() => { setShowAddForm(v => !v); setFormError(''); }}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold text-sm hover:opacity-90 transition"
            >
              {showAddForm ? '✕ Cancel' : '+ Add Holding'}
            </button>
          </div>

          {/* Add Holding Form */}
          {showAddForm && (
            <form onSubmit={handleAdd} className="px-8 py-6 border-b border-gray-100 bg-purple-50">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-end">
                <div>
                  <label className="block text-xs font-semibold text-gray-600 mb-1">Coin</label>
                  <select
                    value={form.crypto_id}
                    onChange={handleCoinSelect}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  >
                    <option value="">Select coin…</option>
                    {KNOWN_COINS.map(c => (
                      <option key={c.id} value={c.id}>{c.symbol} – {c.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-600 mb-1">Amount held</label>
                  <input
                    type="number" step="any" min="0" placeholder="e.g. 0.5"
                    value={form.amount}
                    onChange={e => setForm(f => ({ ...f, amount: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-600 mb-1">Avg buy price (USD)</label>
                  <input
                    type="number" step="any" min="0" placeholder="e.g. 35000"
                    value={form.avg_buy_price}
                    onChange={e => setForm(f => ({ ...f, avg_buy_price: e.target.value }))}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                    required
                  />
                </div>
                <button
                  type="submit"
                  disabled={addHolding.isPending}
                  className="px-6 py-2 rounded-lg bg-purple-600 text-white font-bold text-sm hover:bg-purple-700 disabled:opacity-50 transition"
                >
                  {addHolding.isPending ? 'Adding…' : 'Add'}
                </button>
              </div>
              {formError && <p className="mt-3 text-sm text-red-600 font-medium">{formError}</p>}
            </form>
          )}

          {/* Loading / Error / Empty / Table */}
          {isLoading ? (
            <div className="px-8 py-16 text-center text-gray-400 text-lg">Loading portfolio…</div>
          ) : isError ? (
            <div className="px-8 py-16 text-center text-red-500">Failed to load portfolio. Please sign in.</div>
          ) : holdings.length === 0 ? (
            <div className="px-8 py-16 text-center">
              <p className="text-gray-400 text-lg mb-4">No holdings yet.</p>
              <button
                onClick={() => setShowAddForm(true)}
                className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold text-sm hover:opacity-90 transition"
              >
                + Add your first holding
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gray-50">
                  <tr>
                    {['Asset', 'Amount', 'Current Price', 'Avg Buy Price', 'Market Value', 'Gain/Loss', 'Return', 'Actions'].map(col => (
                      <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {holdings.map((h: PortfolioHolding) => {
                    const currentPrice = (pricesData as Record<string, { usd: number }>)?.[h.crypto_id]?.usd ?? 0;
                    const marketValue = h.amount * currentPrice;
                    const costBasis = h.amount * h.avg_buy_price;
                    const gainLoss = marketValue - costBasis;
                    const gainLossPct = costBasis > 0 ? (gainLoss / costBasis) * 100 : 0;
                    const isEditing = editingId === h.id;

                    return (
                      <tr key={h.id} className="hover:bg-purple-50 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center space-x-3">
                            <CryptoIcon cryptoId={h.crypto_id} size="md" />
                            <div>
                              <div className="text-base font-black text-gray-900">{h.crypto_name}</div>
                              <div className="text-sm font-bold text-gray-500">{h.crypto_symbol}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-black text-gray-900">
                          {isEditing ? (
                            <input type="number" step="any" min="0" value={editForm.amount}
                              onChange={e => setEditForm(f => ({ ...f, amount: e.target.value }))}
                              className="w-24 border border-purple-300 rounded px-2 py-1 text-sm"
                            />
                          ) : `${h.amount.toLocaleString()} ${h.crypto_symbol}`}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-black text-gray-900">
                          {currentPrice > 0 ? fmt(currentPrice) : '—'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-black text-gray-500">
                          {isEditing ? (
                            <input type="number" step="any" min="0" value={editForm.avg_buy_price}
                              onChange={e => setEditForm(f => ({ ...f, avg_buy_price: e.target.value }))}
                              className="w-28 border border-purple-300 rounded px-2 py-1 text-sm"
                            />
                          ) : fmt(h.avg_buy_price)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-black text-gray-900">
                          {currentPrice > 0 ? fmt(marketValue) : '—'}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-black ${gainLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {currentPrice > 0 ? fmt(gainLoss) : '—'}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm font-black ${gainLossPct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {currentPrice > 0 ? fmtPct(gainLossPct) : '—'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex gap-2">
                            {isEditing ? (
                              <>
                                <button onClick={() => handleUpdate(h.id)} disabled={updateHolding.isPending}
                                  className="px-3 py-1 text-xs font-bold rounded-lg bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 transition">
                                  Save
                                </button>
                                <button onClick={() => setEditingId(null)}
                                  className="px-3 py-1 text-xs font-bold rounded-lg bg-gray-200 text-gray-700 hover:bg-gray-300 transition">
                                  Cancel
                                </button>
                              </>
                            ) : (
                              <>
                                <button onClick={() => startEdit(h)}
                                  className="px-3 py-1 text-xs font-bold rounded-lg bg-purple-100 text-purple-700 hover:bg-purple-200 transition">
                                  Edit
                                </button>
                                <button onClick={() => deleteHolding.mutate(h.id)} disabled={deleteHolding.isPending}
                                  className="px-3 py-1 text-xs font-bold rounded-lg bg-red-100 text-red-600 hover:bg-red-200 disabled:opacity-50 transition">
                                  Remove
                                </button>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default PortfolioPage;
