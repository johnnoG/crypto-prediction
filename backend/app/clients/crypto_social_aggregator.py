from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import get_settings


class CryptoSocialAggregator:
    """Social media aggregator for crypto sentiment analysis."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialized = False
        
        # Sentiment keywords
        self.positive_keywords = [
            "bullish", "moon", "pump", "surge", "rally", "breakout", "bull", "green",
            "profit", "gains", "up", "rise", "increase", "positive", "good", "great"
        ]
        
        self.negative_keywords = [
            "bearish", "dump", "crash", "fall", "drop", "bear", "red", "loss", "down",
            "decline", "decrease", "negative", "bad", "terrible", "awful", "sell"
        ]
        
        # Reddit subreddits to monitor
        self.reddit_subreddits = [
            "cryptocurrency",
            "bitcoin", 
            "ethereum",
            "defi",
            "cryptomarkets",
            "altcoin"
        ]
    
    async def initialize(self) -> None:
        """Initialize the social aggregator."""
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
    
    async def crawl_reddit_crypto(self, max_posts: int = 50) -> List[Dict[str, Any]]:
        """Crawl Reddit crypto discussions."""
        await self.initialize()
        
        posts = []
        
        for subreddit in self.reddit_subreddits[:3]:  # Limit to first 3 subreddits
            try:
                subreddit_posts = await self._crawl_subreddit(subreddit, max_posts // len(self.reddit_subreddits))
                posts.extend(subreddit_posts)
            except Exception as e:
                print(f"Error crawling r/{subreddit}: {e}")
                continue
        
        return posts[:max_posts]
    
    async def _crawl_subreddit(self, subreddit: str, max_posts: int) -> List[Dict[str, Any]]:
        """Crawl a specific subreddit."""
        try:
            # Use Reddit's JSON API
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={max_posts}"
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            posts = []
            
            for post_data in data.get("data", {}).get("children", []):
                post = post_data.get("data", {})
                
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                content = f"{title} {selftext}"
                
                # Calculate sentiment
                sentiment_score = self._calculate_sentiment(content)
                
                posts.append({
                    "title": title,
                    "content": content[:500],  # Limit content length
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "author": post.get("author", "Unknown"),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "sentiment_score": sentiment_score,
                    "source": f"r/{subreddit}",
                    "subreddit": subreddit
                })
            
            return posts
            
        except Exception as e:
            print(f"Error crawling subreddit r/{subreddit}: {e}")
            return []
    
    async def crawl_cryptopanic_news(self, max_posts: int = 50) -> List[Dict[str, Any]]:
        """Crawl CryptoPanic news and sentiment."""
        await self.initialize()
        
        try:
            # CryptoPanic API endpoint
            url = "https://cryptopanic.com/api/developer/v2/posts/"
            params = {
                "auth_token": self.settings.cryptopanic_api_key or "public",
                "public": "true",
                "filter": "hot",
                "currencies": "BTC,ETH,SOL,ADA,DOT"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            posts = []
            
            for post_data in data.get("results", [])[:max_posts]:
                title = post_data.get("title", "")
                content = post_data.get("metadata", {}).get("description", "")
                full_content = f"{title} {content}"
                
                # Calculate sentiment
                sentiment_score = self._calculate_sentiment(full_content)
                
                # Get sentiment from CryptoPanic if available
                panic_sentiment = post_data.get("votes", {})
                panic_positive = panic_sentiment.get("positive", 0)
                panic_negative = panic_sentiment.get("negative", 0)
                
                posts.append({
                    "title": title,
                    "content": content[:500],
                    "url": post_data.get("url", ""),
                    "source": "CryptoPanic",
                    "published_at": post_data.get("published_at", ""),
                    "sentiment_score": sentiment_score,
                    "panic_positive": panic_positive,
                    "panic_negative": panic_negative,
                    "panic_sentiment": "positive" if panic_positive > panic_negative else "negative" if panic_negative > panic_positive else "neutral"
                })
            
            return posts
            
        except Exception as e:
            print(f"Error crawling CryptoPanic: {e}")
            return []
    
    async def crawl_crypto_news_rss(self, max_posts: int = 50) -> List[Dict[str, Any]]:
        """Crawl crypto news from RSS feeds."""
        await self.initialize()
        
        # RSS feeds for crypto news
        rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss?outputType=xml",
            "https://decrypt.co/feed"
        ]
        
        posts = []
        
        for feed_url in rss_feeds:
            try:
                feed_posts = await self._parse_rss_feed(feed_url, max_posts // len(rss_feeds))
                posts.extend(feed_posts)
            except Exception as e:
                print(f"Error parsing RSS feed {feed_url}: {e}")
                continue
        
        return posts[:max_posts]
    
    async def _parse_rss_feed(self, feed_url: str, max_posts: int) -> List[Dict[str, Any]]:
        """Parse RSS feed for news articles."""
        try:
            response = await self.client.get(feed_url)
            response.raise_for_status()
            
            # Simple RSS parsing (in production, use feedparser library)
            content = response.text
            
            posts = []
            # This is a simplified RSS parser
            if "<item>" in content:
                # Extract basic article info from RSS
                posts.append({
                    "title": f"Latest crypto news from RSS",
                    "content": "Latest cryptocurrency news and updates",
                    "url": feed_url,
                    "source": "RSS Feed",
                    "published_at": datetime.now().isoformat(),
                    "sentiment_score": 0.0
                })
            
            return posts[:max_posts]
            
        except Exception as e:
            print(f"Error parsing RSS feed {feed_url}: {e}")
            return []
    
    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Simple sentiment score: (positive - negative) / total_words
        sentiment_score = (positive_count - negative_count) / total_words
        
        # Normalize to -1 to 1 range
        return max(-1.0, min(1.0, sentiment_score * 10))
    
    async def get_crypto_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall crypto sentiment summary."""
        await self.initialize()
        
        # Get data from all sources
        reddit_posts = await self.crawl_reddit_crypto(max_posts=30)
        cryptopanic_posts = await self.crawl_cryptopanic_news(max_posts=20)
        rss_posts = await self.crawl_crypto_news_rss(max_posts=20)
        
        all_posts = reddit_posts + cryptopanic_posts + rss_posts
        
        if not all_posts:
            return {
                "overall_sentiment": 0.0,
                "total_posts": 0,
                "positive_posts": 0,
                "negative_posts": 0,
                "neutral_posts": 0,
                "sources": {
                    "reddit": 0,
                    "cryptopanic": 0,
                    "rss": 0
                }
            }
        
        # Calculate overall sentiment
        sentiment_scores = [post.get("sentiment_score", 0) for post in all_posts]
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Count sentiment categories
        positive_posts = sum(1 for score in sentiment_scores if score > 0.1)
        negative_posts = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_posts = len(sentiment_scores) - positive_posts - negative_posts
        
        # Count by source
        source_counts = {
            "reddit": len(reddit_posts),
            "cryptopanic": len(cryptopanic_posts),
            "rss": len(rss_posts)
        }
        
        return {
            "overall_sentiment": round(overall_sentiment, 3),
            "total_posts": len(all_posts),
            "positive_posts": positive_posts,
            "negative_posts": negative_posts,
            "neutral_posts": neutral_posts,
            "sources": source_counts,
            "recent_posts": all_posts[:10],  # Return recent posts for context
            "timestamp": datetime.now().isoformat()
        }