import { useState, type ReactElement } from 'react';

interface CryptoIconProps {
  cryptoId: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

const sizeClasses = {
  xs: 'w-5 h-5',
  sm: 'w-8 h-8',
  md: 'w-12 h-12',
  lg: 'w-16 h-16',
  xl: 'w-20 h-20',
};

// CoinGecko CDN - Most reliable source for crypto icons
// Format: https://assets.coingecko.com/coins/images/{image_id}/{size}/{name}.png
// We'll use CoinGecko's public CDN with proper image IDs

// CoinGecko image IDs (these are the actual IDs from CoinGecko's database)
const COINGECKO_IMAGE_IDS: Record<string, number> = {
  'bitcoin': 1,
  'ethereum': 279,
  'tether': 825,
  'binancecoin': 1839,
  'solana': 4128,
  'ripple': 52,
  'usd-coin': 3408,
  'cardano': 975,
  'avalanche-2': 12559,
  'dogecoin': 5,
  'polkadot': 12171,
  'matic-network': 4713,
  'chainlink': 877,
  'litecoin': 2,
  'uniswap': 12504,
  'bitcoin-cash': 1831,
  'cosmos': 3794,
  'stellar': 512,
  'ethereum-classic': 1321,
  'filecoin': 4890,
  'monero': 328,
  'aave': 7278,
  'algorand': 4030,
  'the-sandbox': 12129,
  'axie-infinity': 11636,
  'decentraland': 9168,
  'theta-token': 2416,
  'fantom': 3513,
  'near': 11165,
  'optimism': 11840,
  'arbitrum': 11841,
};

// Get CoinGecko icon URL
const getCoinGeckoIconUrl = (cryptoId: string, size: 'small' | 'large' = 'small'): string => {
  const imageId = COINGECKO_IMAGE_IDS[cryptoId];
  if (imageId) {
    // CoinGecko CDN format
    return `https://assets.coingecko.com/coins/images/${imageId}/${size}/${cryptoId}.png`;
  }
  // Fallback: try using cryptoicons.org
  const symbol = cryptoId.replace(/-/g, '').toLowerCase();
  return `https://cryptoicons.org/api/icon/${symbol}/32`;
};

// Fallback SVG icons for when images fail to load
const FallbackIcons: Record<string, ReactElement> = {
  bitcoin: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm7.189-17.98c.314-2.096-1.283-3.223-3.465-3.975l.708-2.84-1.728-.43-.69 2.765c-.454-.114-.92-.22-1.385-.326l.695-2.783L15.596 6l-.708 2.839c-.376-.086-.746-.17-1.104-.26l.002-.009-2.384-.595-.46 1.846s1.283.294 1.256.312c.7.175.826.638.805 1.006l-.806 3.235c.048.012.11.03.18.057l-.183-.045-1.13 4.532c-.086.212-.303.531-.793.41.018.025-1.256-.313-1.256-.313l-.858 1.978 2.25.561c.418.105.828.215 1.231.318l-.715 2.872 1.727.43.708-2.84c.472.127.93.245 1.378.357l-.706 2.828 1.728.43.715-2.866c2.948.558 5.164.333 6.097-2.333.752-2.146-.037-3.385-1.588-4.192 1.13-.26 1.98-1.003 2.207-2.538zm-3.95 5.538c-.533 2.147-4.148.986-5.32.695l.95-3.805c1.172.293 4.929.872 4.37 3.11zm.535-5.569c-.487 1.953-3.495.96-4.47.717l.86-3.45c.975.243 4.118.696 3.61 2.733z" />
    </svg>
  ),
  ethereum: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M15.927 23.959l-9.823-5.797 9.817 13.839 9.828-13.839-9.828 5.797zM16.073 0l-9.819 16.297 9.819 5.807 9.823-5.801z" />
    </svg>
  ),
  tether: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-1.333-5.333V22.4c-1.6-.133-5.333-.8-5.333-3.733 0-2.133 1.867-3.2 5.333-3.733V9.333h2.666v5.6c3.467.533 5.333 1.6 5.333 3.733 0 2.933-3.733 3.6-5.333 3.733v4.267zm0-7.467V14.4c-2.4.267-4 .8-4 1.867 0 .8.8 1.333 2.667 1.6zm2.666 0c1.867-.267 2.667-.8 2.667-1.6 0-1.067-1.6-1.6-4-1.867v2.667z" />
    </svg>
  ),
  binancecoin: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-3.884-17.596L16 10.52l3.886 3.886 2.26-2.26L16 6l-6.144 6.144 2.26 2.26zM6 16l2.26 2.26L10.52 16l-2.26-2.26L6 16zm6.116 1.596l-2.263 2.257.003.003L16 26l6.146-6.146v-.001l-2.26-2.26L16 21.48l-3.884-3.884zm7.77-5.335l2.26 2.26L24.407 16l-2.26 2.26-2.26-2.26 2.26-2.26z" />
    </svg>
  ),
  solana: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M26.7 12.88c.22-.2.52-.32.84-.32H31c.66 0 1 .79.54 1.27l-4.46 4.45c-.22.2-.52.32-.84.32H22.78c-.66 0-1-.79-.54-1.27zm0 8.57c.22-.2.52-.32.84-.32H31c.66 0 1 .79.54 1.27l-4.46 4.45c-.22.2-.52.32-.84.32H22.78c-.66 0-1-.79-.54-1.27zM4.7 10.55C4.92 10.35 5.22 10.23 5.54 10.23H9c.66 0 1 .79.54 1.27l-4.46 4.45c-.22.2-.52.32-.84.32H.78c-.66 0-1-.79-.54-1.27z" />
    </svg>
  ),
  ripple: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm6.808 7.255c-3.473-2.678-8.143-2.678-11.616 0l-2.384 1.836c-.402.31-.402.93 0 1.24l2.384 1.836c3.473 2.678 8.143 2.678 11.616 0l2.384-1.836c.402-.31.402-.93 0-1.24zm-2.384 11.836l-2.384 1.836c-3.473 2.678-8.143 2.678-11.616 0l-2.384-1.836c-.402-.31-.402-.93 0-1.24l2.384-1.836c3.473-2.678 8.143-2.678 11.616 0l2.384 1.836c.402.31.402.93 0 1.24z" />
    </svg>
  ),
  'usd-coin': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm0-2.667c-4.418 0-8-3.582-8-8s3.582-8 8-8 8 3.582 8 8-3.582 8-8 8zm0-2.667c2.946 0 5.333-2.388 5.333-5.333S18.946 10.667 16 10.667 10.667 13.054 10.667 16 13.054 21.333 16 21.333z" />
    </svg>
  ),
  cardano: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M13.735 17.64l-.555.908c.185.113.407.18.647.18s.462-.067.647-.18l-.555-.908a.828.828 0 01-.185.024.828.828 0 01-.185-.024zm4.53-5.33a5.594 5.594 0 001.55-5.308L19.257 8a4.767 4.767 0 00-1.328 4.548l.555.908c.185.113.407.18.647.18s.462-.067.647-.18l.555-.908a4.767 4.767 0 00-1.328-4.548L18.45 7c.738.454 1.326 1.092 1.706 1.855l.556.908a.828.828 0 01-.37.277l.555.908a1.655 1.655 0 00.74-.555l.556-.908c-.38-.763-.968-1.4-1.706-1.855L19.93 8a5.594 5.594 0 00-1.55 5.308l-.555.908c-.185-.113-.407-.18-.647-.18s-.462.067-.647.18l-.555-.908a5.594 5.594 0 00-1.55-5.308l.556-.908c.738.454 1.326 1.092 1.706 1.855l.556.908a.828.828 0 01-.37.277l.555.908a1.655 1.655 0 00.74-.555l.556-.908c-.38-.763-.968-1.4-1.706-1.855L16.47 8c.738.454 1.326 1.092 1.706 1.855zm-2.446 7.46a4.767 4.767 0 001.328 4.548l-.555.908c-.738-.454-1.326-1.092-1.706-1.855l-.556-.908a.828.828 0 01.37-.277l-.555-.908a1.655 1.655 0 00-.74.555l-.556.908c.38.763.968 1.4 1.706 1.855l.555.908a5.594 5.594 0 001.55-5.308l.555-.908c.185.113.407.18.647.18s.462-.067.647-.18l.555.908a5.594 5.594 0 001.55 5.308l-.555.908c-.738-.454-1.326-1.092-1.706-1.855l-.556-.908a.828.828 0 01.37-.277l-.555-.908a1.655 1.655 0 00-.74.555l-.556.908c.38.763.968 1.4 1.706 1.855l.555.908c-.738-.454-1.326-1.092-1.706-1.855z" />
    </svg>
  ),
  'avalanche-2': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-2.667-8l-4-6.667h8l-4 6.667zm0-10.667l-4-6.666h8l-4 6.666zm5.334 0l4 6.666h-8l4-6.666z" />
    </svg>
  ),
  dogecoin: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-26.667c-5.891 0-10.667 4.776-10.667 10.667S10.109 26.667 16 26.667 26.667 21.891 26.667 16 21.891 5.333 16 5.333zm-1.333 5.334h2.666v5.333h2.667v2.667h-2.667v5.333h-2.666v-5.333H12v-2.667h2.667z" />
    </svg>
  ),
  polkadot: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm0 6.154c-1.473 0-2.667 1.194-2.667 2.667 0 1.473 1.194 2.667 2.667 2.667 1.473 0 2.667-1.194 2.667-2.667 0-1.473-1.194-2.667-2.667-2.667zm0 6.154a2.667 2.667 0 110 5.333 2.667 2.667 0 010-5.333zm5.333 3.487c-1.473 0-2.667 1.194-2.667 2.667 0 1.473 1.194 2.667 2.667 2.667 1.473 0 2.667-1.194 2.667-2.667 0-1.473-1.194-2.667-2.667-2.667zm-10.666 0c-1.473 0-2.667 1.194-2.667 2.667 0 1.473 1.194 2.667 2.667 2.667 1.473 0 2.667-1.194 2.667-2.667 0-1.473-1.194-2.667-2.667-2.667zm5.333 4.513c-1.473 0-2.667 1.194-2.667 2.667 0 1.473 1.194 2.667 2.667 2.667 1.473 0 2.667-1.194 2.667-2.667 0-1.473-1.194-2.667-2.667-2.667z" />
    </svg>
  ),
  'matic-network': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-2.667-8l-4-6.667h8l-4 6.667zm0-10.667l-4-6.666h8l-4 6.666zm5.334 0l4 6.666h-8l4-6.666zm0 10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  chainlink: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm0 6.4c-5.302 0-9.6 4.298-9.6 9.6s4.298 9.6 9.6 9.6 9.6-4.298 9.6-9.6-4.298-9.6-9.6-9.6zm0 2.133c4.118 0 7.467 3.349 7.467 7.467S20.118 23.467 16 23.467 8.533 20.118 8.533 16s3.349-7.467 7.467-7.467z" />
    </svg>
  ),
  litecoin: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-1.333-8l8-2.667L20 18.667l-5.333 1.6V12l-2.667.8v8.267z" />
    </svg>
  ),
  uniswap: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm-2.667-8l4-4 4 4h-8zm0-5.334l4-4 4 4h-8z" />
    </svg>
  ),
  'bitcoin-cash': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-1.333-8l8-2.667L20 18.667l-5.333 1.6V12l-2.667.8v8.267zm-2.667-10.667h2.667v2.667h-2.667zm0 5.334h2.667v2.667h-2.667z" />
    </svg>
  ),
  cosmos: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm0 6.4c-5.302 0-9.6 4.298-9.6 9.6s4.298 9.6 9.6 9.6 9.6-4.298 9.6-9.6-4.298-9.6-9.6-9.6zm0 2.133c4.118 0 7.467 3.349 7.467 7.467S20.118 23.467 16 23.467 8.533 20.118 8.533 16s3.349-7.467 7.467-7.467zm-4.8 6.4h9.6v2.133h-9.6z" />
    </svg>
  ),
  stellar: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm0 6.4c-5.302 0-9.6 4.298-9.6 9.6s4.298 9.6 9.6 9.6 9.6-4.298 9.6-9.6-4.298-9.6-9.6-9.6zm0 2.133c4.118 0 7.467 3.349 7.467 7.467S20.118 23.467 16 23.467 8.533 20.118 8.533 16s3.349-7.467 7.467-7.467zm-2.4 4.8l4.8 4.8-4.8 4.8v-9.6zm4.8 0l4.8 4.8-4.8 4.8v-9.6z" />
    </svg>
  ),
  'ethereum-classic': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M15.927 23.959l-9.823-5.797 9.817 13.839 9.828-13.839-9.828 5.797zM16.073 0l-9.819 16.297 9.819 5.807 9.823-5.801z" />
    </svg>
  ),
  filecoin: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-2.667-8l-4-6.667h8l-4 6.667zm0-10.667l-4-6.666h8l-4 6.666zm5.334 0l4 6.666h-8l4-6.666zm0 10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  monero: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm-2.667-8l4-4 4 4h-8zm0-5.334l4-4 4 4h-8z" />
    </svg>
  ),
  aave: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm0 6.4c-5.302 0-9.6 4.298-9.6 9.6s4.298 9.6 9.6 9.6 9.6-4.298 9.6-9.6-4.298-9.6-9.6-9.6zm0 2.133c4.118 0 7.467 3.349 7.467 7.467S20.118 23.467 16 23.467 8.533 20.118 8.533 16s3.349-7.467 7.467-7.467zm-2.4 4.8h4.8v9.6h-4.8z" />
    </svg>
  ),
  algorand: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-2.667-8l-4-6.667h8l-4 6.667zm0-10.667l-4-6.666h8l-4 6.666zm5.334 0l4 6.666h-8l4-6.666zm0 10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  'the-sandbox': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm-2.667 8l-4 6.667h8l-4-6.667zm0 10.667l-4 6.666h8l-4-6.666zm5.334 0l4 6.666h-8l4-6.666zm0-10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  'axie-infinity': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm-2.667-8l4-4 4 4h-8zm0-5.334l4-4 4 4h-8z" />
    </svg>
  ),
  decentraland: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm-2.667 8l-4 6.667h8l-4-6.667zm0 10.667l-4 6.666h8l-4-6.666zm5.334 0l4 6.666h-8l4-6.666zm0-10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  'theta-token': (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm-2.667-8l4-4 4 4h-8zm0-5.334l4-4 4 4h-8z" />
    </svg>
  ),
  fantom: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm-2.667 8l-4 6.667h8l-4-6.667zm0 10.667l-4 6.666h8l-4-6.666zm5.334 0l4 6.666h-8l4-6.666zm0-10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  near: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm-2.667-8l4-4 4 4h-8zm0-5.334l4-4 4 4h-8z" />
    </svg>
  ),
  optimism: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 0c8.837 0 16 7.163 16 16s-7.163 16-16 16S0 24.837 0 16 7.163 0 16 0zm-2.667 8l-4 6.667h8l-4-6.667zm0 10.667l-4 6.666h8l-4-6.666zm5.334 0l4 6.666h-8l4-6.666zm0-10.667l4 6.667h-8l4-6.667z" />
    </svg>
  ),
  arbitrum: (
    <svg viewBox="0 0 32 32" fill="currentColor" className="w-full h-full">
      <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm0-5.333c5.891 0 10.667-4.776 10.667-10.667S21.891 5.333 16 5.333 5.333 10.109 5.333 16 10.109 26.667 16 26.667zm-2.667-8l4-4 4 4h-8zm0-5.334l4-4 4 4h-8z" />
    </svg>
  ),
};

// Get gradient style for each crypto
const getCryptoGradient = (cryptoId: string): string => {
  const gradients: Record<string, string> = {
    'bitcoin': 'from-orange-400 to-orange-600',
    'ethereum': 'from-blue-400 to-blue-600',
    'tether': 'from-green-400 to-green-600',
    'binancecoin': 'from-yellow-400 to-yellow-600',
    'solana': 'from-purple-400 to-purple-600',
    'ripple': 'from-blue-500 to-cyan-600',
    'usd-coin': 'from-blue-400 to-blue-600',
    'cardano': 'from-blue-500 to-indigo-600',
    'avalanche-2': 'from-red-400 to-red-600',
    'dogecoin': 'from-yellow-500 to-yellow-700',
    'polkadot': 'from-pink-400 to-pink-600',
    'matic-network': 'from-purple-500 to-purple-700',
    'chainlink': 'from-blue-400 to-blue-600',
    'litecoin': 'from-gray-400 to-gray-600',
    'uniswap': 'from-pink-400 to-pink-600',
    'bitcoin-cash': 'from-orange-500 to-orange-700',
    'cosmos': 'from-indigo-400 to-indigo-600',
    'stellar': 'from-purple-400 to-purple-600',
    'ethereum-classic': 'from-gray-500 to-gray-700',
    'filecoin': 'from-blue-400 to-blue-600',
    'monero': 'from-orange-500 to-orange-700',
    'aave': 'from-purple-400 to-purple-600',
    'algorand': 'from-gray-400 to-gray-600',
    'the-sandbox': 'from-yellow-400 to-yellow-600',
    'axie-infinity': 'from-purple-400 to-purple-600',
    'decentraland': 'from-pink-400 to-pink-600',
    'theta-token': 'from-blue-400 to-blue-600',
    'fantom': 'from-blue-400 to-blue-600',
    'near': 'from-black to-gray-800',
    'optimism': 'from-red-400 to-red-600',
    'arbitrum': 'from-blue-400 to-blue-600',
  };
  
  return gradients[cryptoId] || 'from-gray-400 to-gray-600';
};

function CryptoIcon({ cryptoId, size = 'md', className = '' }: CryptoIconProps) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const sizeClass = sizeClasses[size];
  const gradient = getCryptoGradient(cryptoId);
  
  // Use CoinGecko CDN with appropriate size
  const iconSize = size === 'sm' || size === 'md' ? 'small' : 'large';
  const iconUrl = getCoinGeckoIconUrl(cryptoId, iconSize);
  const fallbackIcon = FallbackIcons[cryptoId];

  // If image failed to load, show fallback
  if (imageError) {
    return (
      <div
        className={`${sizeClass} rounded-2xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg text-white p-2 transform hover:scale-110 transition-transform duration-300 ${className}`}
      >
        {fallbackIcon || (
          <span className="text-sm font-bold uppercase">
            {cryptoId.charAt(0).toUpperCase()}
          </span>
        )}
      </div>
    );
  }

  return (
    <div
      className={`${sizeClass} rounded-2xl bg-gradient-to-br ${gradient} flex items-center justify-center shadow-lg overflow-hidden transform hover:scale-110 transition-transform duration-300 ${className} ${!imageLoaded ? 'bg-opacity-50' : ''}`}
    >
      <img
        src={iconUrl}
        alt={cryptoId}
        className={`w-full h-full object-contain p-1 ${imageLoaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}
        onLoad={() => setImageLoaded(true)}
        onError={() => {
          setImageError(true);
          setImageLoaded(false);
        }}
        loading="lazy"
      />
      {!imageLoaded && !imageError && (
        <div className="absolute inset-0 flex items-center justify-center">
          {fallbackIcon || (
            <span className="text-xs font-bold uppercase text-white/50">
              {cryptoId.charAt(0).toUpperCase()}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default CryptoIcon;

