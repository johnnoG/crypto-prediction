from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx


class CryptoSocialClient:
    """Free crypto social media sentiment crawler."""
    
    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        
        # Free social media APIs and sources
        self.sources = {
            "reddit": {
                "base_url": "https://www.reddit.com/r/cryptocurrency/hot.json",
                "fallback_urls": [
                    "https://www.reddit.com/r/bitcoin/hot.json",
                    "https://www.reddit.com/r/ethereum/hot.json",
                    "https://www.reddit.com/r/cryptomarkets/hot.json"
                ]
            },
            "cryptopanic": {
                "base_url": "https://cryptopanic.com/api/v1/posts/",
                "params": {
                    "auth_token": "free",  # Free tier
                    "public": "true",
                    "filter": "hot"
                }
            }
        }
        
        # Crypto-related keywords for sentiment analysis
        self.crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
            "blockchain", "defi", "nft", "altcoin", "trading", "hodl",
            "bull", "bear", "pump", "dump", "moon", "lambo"
        ]

    async def close(self) -> None:
        await self._client.aclose()

    def _extract_sentiment_keywords(self, text: str) -> Dict[str, int]:
        """Extract sentiment indicators from text."""
        text_lower = text.lower()
        
        positive_words = [
            "bull", "bullish", "moon", "pump", "rise", "up", "gain", "profit",
            "buy", "hodl", "diamond", "hands", "lambo", "green", "surge"
        ]
        
        negative_words = [
            "bear", "bearish", "dump", "crash", "down", "loss", "sell",
            "panic", "fud", "red", "drop", "decline", "rekt"
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": max(0, len(text.split()) - positive_count - negative_count)
        }

    def _calculate_sentiment_score(self, sentiment_data: Dict[str, int]) -> float:
        """Calculate sentiment score from -1 (very negative) to 1 (very positive)."""
        total = sum(sentiment_data.values())
        if total == 0:
            return 0.0
        
        positive_ratio = sentiment_data["positive"] / total
        negative_ratio = sentiment_data["negative"] / total
        
        return positive_ratio - negative_ratio

    def _normalize_social_post(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Normalize social media post data."""
        text = data.get("text", data.get("title", data.get("content", "")))
        url = data.get("url", data.get("link", ""))
        
        sentiment_data = self._extract_sentiment_keywords(text)
        sentiment_score = self._calculate_sentiment_score(sentiment_data)
        
        return {
            "url": url,
            "title": data.get("title", text[:100] + "..." if len(text) > 100 else text),
            "content": text,
            "author": data.get("author", "Unknown"),
            "source": source,
            "published_at": data.get("created_at", datetime.now(timezone.utc).isoformat()),
            "sentiment_score": sentiment_score,
            "sentiment_data": sentiment_data,
            "engagement": {
                "upvotes": data.get("upvotes", data.get("score", 0)),
                "comments": data.get("comments", data.get("num_comments", 0)),
                "shares": data.get("shares", 0)
            },
            "url_hash": hashlib.sha256(url.encode()).hexdigest() if url else hashlib.sha256(text.encode()).hexdigest(),
            "crawled_at": datetime.now(timezone.utc).isoformat()
        }

    async def crawl_reddit_crypto(self, max_posts: int = 20) -> List[Dict[str, Any]]:
        """Crawl Reddit crypto discussions."""
        posts = []
        
        try:
            # Try main crypto subreddit first
            response = await self._client.get(self.sources["reddit"]["base_url"])
            response.raise_for_status()
            
            data = response.json()
            reddit_posts = data.get("data", {}).get("children", [])
            
            for post_data in reddit_posts[:max_posts]:
                post = post_data.get("data", {})
                normalized = self._normalize_social_post({
                    "title": post.get("title", ""),
                    "text": post.get("selftext", ""),
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "author": post.get("author", "Unknown"),
                    "upvotes": post.get("ups", 0),
                    "comments": post.get("num_comments", 0),
                    "created_at": datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc).isoformat()
                }, "reddit")
                posts.append(normalized)
                
        except Exception as e:
            print(f"Reddit crawl failed: {e}")
            
        return posts

    async def crawl_cryptopanic_news(self, max_posts: int = 20) -> List[Dict[str, Any]]:
        """Crawl CryptoPanic news and sentiment."""
        posts = []
        
        try:
            response = await self._client.get(
                self.sources["cryptopanic"]["base_url"],
                params=self.sources["cryptopanic"]["params"]
            )
            response.raise_for_status()
            
            data = response.json()
            news_posts = data.get("results", [])
            
            for post_data in news_posts[:max_posts]:
                normalized = self._normalize_social_post({
                    "title": post_data.get("title", ""),
                    "content": post_data.get("description", ""),
                    "url": post_data.get("url", ""),
                    "author": post_data.get("source", {}).get("title", "Unknown"),
                    "created_at": post_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "upvotes": post_data.get("votes", {}).get("positive", 0),
                    "comments": post_data.get("votes", {}).get("negative", 0)
                }, "cryptopanic")
                posts.append(normalized)
                
        except Exception as e:
            print(f"CryptoPanic crawl failed: {e}")
            
        return posts

    async def crawl_all_social_sources(self, max_posts_per_source: int = 10) -> List[Dict[str, Any]]:
        """Crawl all social media sources."""
        all_posts = []
        
        # Crawl Reddit
        reddit_posts = await self.crawl_reddit_crypto(max_posts_per_source)
        all_posts.extend(reddit_posts)
        
        # Crawl CryptoPanic
        cryptopanic_posts = await self.crawl_cryptopanic_news(max_posts_per_source)
        all_posts.extend(cryptopanic_posts)
        
        return all_posts

    async def get_crypto_sentiment_summary(self) -> Dict[str, Any]:
        """Get overall crypto sentiment summary."""
        posts = await self.crawl_all_social_sources(max_posts_per_source=5)
        
        if not posts:
            return {
                "overall_sentiment": 0.0,
                "total_posts": 0,
                "positive_posts": 0,
                "negative_posts": 0,
                "neutral_posts": 0,
                "sources": {}
            }
        
        total_sentiment = 0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        source_sentiments = {}
        
        for post in posts:
            sentiment = post.get("sentiment_score", 0)
            total_sentiment += sentiment
            source = post.get("source", "unknown")
            
            if sentiment > 0.1:
                positive_count += 1
            elif sentiment < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
            
            if source not in source_sentiments:
                source_sentiments[source] = {"total": 0, "count": 0}
            source_sentiments[source]["total"] += sentiment
            source_sentiments[source]["count"] += 1
        
        # Calculate average sentiment by source
        for source in source_sentiments:
            if source_sentiments[source]["count"] > 0:
                source_sentiments[source]["average"] = (
                    source_sentiments[source]["total"] / source_sentiments[source]["count"]
                )
        
        return {
            "overall_sentiment": total_sentiment / len(posts) if posts else 0,
            "total_posts": len(posts),
            "positive_posts": positive_count,
            "negative_posts": negative_count,
            "neutral_posts": neutral_count,
            "sources": source_sentiments,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

