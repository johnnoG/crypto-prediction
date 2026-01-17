from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from config import get_settings


class TwitterCryptoClient:
    """Twitter crypto client using Nitter (free Twitter alternative)."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialized = False
        
        # Nitter instances (free Twitter alternatives)
        self.nitter_instances = [
            "https://nitter.net",
            "https://nitter.1d4.us", 
            "https://nitter.kavin.rocks",
            "https://nitter.unixfox.eu"
        ]
        
        # Crypto-related hashtags and accounts
        self.crypto_hashtags = [
            "#Bitcoin", "#BTC", "#Ethereum", "#ETH", "#Solana", "#SOL",
            "#Cardano", "#ADA", "#Polkadot", "#DOT", "#Chainlink", "#LINK",
            "#crypto", "#cryptocurrency", "#DeFi", "#NFT", "#Web3"
        ]
        
        self.crypto_accounts = [
            "elonmusk", "VitalikButerin", "naval", "APompliano", "michael_saylor",
            "cz_binance", "justinsuntron", "brian_armstrong", "rogerkver"
        ]
    
    async def initialize(self) -> None:
        """Initialize the Twitter client."""
        if self._initialized:
            return
            
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        self._initialized = True
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
    
    async def crawl_twitter_hashtag(self, hashtag: str, max_tweets: int = 20) -> List[Dict[str, Any]]:
        """Crawl tweets for a specific hashtag."""
        await self.initialize()
        
        tweets = []
        
        for nitter_instance in self.nitter_instances:
            try:
                instance_tweets = await self._crawl_hashtag_from_nitter(nitter_instance, hashtag, max_tweets)
                tweets.extend(instance_tweets)
                
                if len(tweets) >= max_tweets:
                    break
                    
            except Exception as e:
                print(f"Error crawling hashtag {hashtag} from {nitter_instance}: {e}")
                continue
        
        return tweets[:max_tweets]
    
    async def crawl_twitter_account(self, account: str, max_tweets: int = 20) -> List[Dict[str, Any]]:
        """Crawl tweets from a specific account."""
        await self.initialize()
        
        tweets = []
        
        for nitter_instance in self.nitter_instances:
            try:
                instance_tweets = await self._crawl_account_from_nitter(nitter_instance, account, max_tweets)
                tweets.extend(instance_tweets)
                
                if len(tweets) >= max_tweets:
                    break
                    
            except Exception as e:
                print(f"Error crawling account {account} from {nitter_instance}: {e}")
                continue
        
        return tweets[:max_tweets]
    
    async def _crawl_hashtag_from_nitter(self, nitter_instance: str, hashtag: str, max_tweets: int) -> List[Dict[str, Any]]:
        """Crawl hashtag tweets from a Nitter instance."""
        try:
            url = f"{nitter_instance}/search?q={hashtag}&f=tweets"
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            tweets = []
            
            # Find tweet elements (Nitter-specific selectors)
            tweet_elements = soup.find_all('div', class_='tweet-content')
            
            for tweet_element in tweet_elements[:max_tweets]:
                try:
                    # Extract tweet text
                    tweet_text = tweet_element.get_text(strip=True)
                    
                    # Extract metadata
                    tweet_meta = tweet_element.find_parent('div', class_='tweet')
                    if not tweet_meta:
                        continue
                    
                    # Extract author
                    author_element = tweet_meta.find('a', class_='username')
                    author = author_element.get_text(strip=True) if author_element else "Unknown"
                    
                    # Extract timestamp
                    time_element = tweet_meta.find('span', class_='tweet-date')
                    timestamp = time_element.get('title', '') if time_element else datetime.now().isoformat()
                    
                    # Calculate sentiment
                    sentiment_score = self._calculate_sentiment(tweet_text)
                    
                    tweets.append({
                        "text": tweet_text,
                        "author": author,
                        "hashtag": hashtag,
                        "timestamp": timestamp,
                        "sentiment_score": sentiment_score,
                        "source": "Twitter via Nitter",
                        "url": f"{nitter_instance}/search?q={hashtag}"
                    })
                    
                except Exception as e:
                    print(f"Error parsing tweet: {e}")
                    continue
            
            return tweets
            
        except Exception as e:
            print(f"Error crawling hashtag from {nitter_instance}: {e}")
            return []
    
    async def _crawl_account_from_nitter(self, nitter_instance: str, account: str, max_tweets: int) -> List[Dict[str, Any]]:
        """Crawl account tweets from a Nitter instance."""
        try:
            url = f"{nitter_instance}/{account}"
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            tweets = []
            
            # Find tweet elements
            tweet_elements = soup.find_all('div', class_='tweet-content')
            
            for tweet_element in tweet_elements[:max_tweets]:
                try:
                    # Extract tweet text
                    tweet_text = tweet_element.get_text(strip=True)
                    
                    # Extract metadata
                    tweet_meta = tweet_element.find_parent('div', class_='tweet')
                    if not tweet_meta:
                        continue
                    
                    # Extract timestamp
                    time_element = tweet_meta.find('span', class_='tweet-date')
                    timestamp = time_element.get('title', '') if time_element else datetime.now().isoformat()
                    
                    # Calculate sentiment
                    sentiment_score = self._calculate_sentiment(tweet_text)
                    
                    tweets.append({
                        "text": tweet_text,
                        "author": account,
                        "account": account,
                        "timestamp": timestamp,
                        "sentiment_score": sentiment_score,
                        "source": "Twitter via Nitter",
                        "url": f"{nitter_instance}/{account}"
                    })
                    
                except Exception as e:
                    print(f"Error parsing tweet: {e}")
                    continue
            
            return tweets
            
        except Exception as e:
            print(f"Error crawling account from {nitter_instance}: {e}")
            return []
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for tweet text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Simple sentiment keywords
        positive_keywords = [
            "bullish", "moon", "pump", "surge", "rally", "breakout", "bull", "green",
            "profit", "gains", "up", "rise", "increase", "positive", "good", "great",
            "love", "amazing", "awesome", "🚀", "📈", "💎", "hodl"
        ]
        
        negative_keywords = [
            "bearish", "dump", "crash", "fall", "drop", "bear", "red", "loss", "down",
            "decline", "decrease", "negative", "bad", "terrible", "awful", "sell",
            "hate", "worst", "disappointed", "📉", "💸", "😢"
        ]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Simple sentiment score
        sentiment_score = (positive_count - negative_count) / total_words
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment_score * 5))
    
    async def get_crypto_sentiment(self) -> Dict[str, Any]:
        """Get overall crypto sentiment from Twitter."""
        await self.initialize()
        
        all_tweets = []
        
        # Crawl popular crypto hashtags
        for hashtag in self.crypto_hashtags[:5]:  # Limit to first 5 hashtags
            try:
                tweets = await self.crawl_twitter_hashtag(hashtag, max_tweets=10)
                all_tweets.extend(tweets)
            except Exception as e:
                print(f"Error crawling hashtag {hashtag}: {e}")
                continue
        
        # Crawl popular crypto accounts
        for account in self.crypto_accounts[:3]:  # Limit to first 3 accounts
            try:
                tweets = await self.crawl_twitter_account(account, max_tweets=5)
                all_tweets.extend(tweets)
            except Exception as e:
                print(f"Error crawling account {account}: {e}")
                continue
        
        if not all_tweets:
            return {
                "overall_sentiment": 0.0,
                "total_tweets": 0,
                "positive_tweets": 0,
                "negative_tweets": 0,
                "neutral_tweets": 0,
                "recent_tweets": []
            }
        
        # Calculate overall sentiment
        sentiment_scores = [tweet.get("sentiment_score", 0) for tweet in all_tweets]
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Count sentiment categories
        positive_tweets = sum(1 for score in sentiment_scores if score > 0.1)
        negative_tweets = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_tweets = len(sentiment_scores) - positive_tweets - negative_tweets
        
        return {
            "overall_sentiment": round(overall_sentiment, 3),
            "total_tweets": len(all_tweets),
            "positive_tweets": positive_tweets,
            "negative_tweets": negative_tweets,
            "neutral_tweets": neutral_tweets,
            "recent_tweets": all_tweets[:10],  # Return recent tweets for context
            "timestamp": datetime.now().isoformat()
        }