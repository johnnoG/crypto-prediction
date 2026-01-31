from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import get_settings


class CryptoNewsClient:
    """Crypto news client that crawls major crypto news sources."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialized = False
        
        # News sources configuration
        self.news_sources = {
            "coindesk": {
                "name": "CoinDesk",
                "base_url": "https://www.coindesk.com",
                "rss_url": "https://www.coindesk.com/arc/outboundfeeds/rss?outputType=xml",
                "enabled": True
            },
            "cointelegraph": {
                "name": "CoinTelegraph", 
                "base_url": "https://cointelegraph.com",
                "rss_url": "https://cointelegraph.com/rss",
                "enabled": True
            },
            "decrypt": {
                "name": "Decrypt",
                "base_url": "https://decrypt.co",
                "rss_url": "https://decrypt.co/feed",
                "enabled": True
            },
            "theblock": {
                "name": "The Block",
                "base_url": "https://www.theblock.co",
                "rss_url": "https://www.theblock.co/rss.xml",
                "enabled": True
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the news client."""
        if self._initialized:
            return
            
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self._initialized = True
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.aclose()
    
    async def get_all_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get news from all enabled sources."""
        await self.initialize()
        
        all_articles = []
        
        # Get news from each source
        for source_id, source_config in self.news_sources.items():
            if source_config["enabled"]:
                try:
                    articles = await self.get_news_by_source(source_id, limit=limit//len(self.news_sources))
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Error getting news from {source_id}: {e}")
                    continue
        
        # Sort by published date and limit
        all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return all_articles[:limit]
    
    async def get_news_by_source(self, source: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get news from a specific source."""
        await self.initialize()
        
        if source not in self.news_sources:
            return []
        
        source_config = self.news_sources[source]
        
        try:
            # Try RSS first
            articles = await self._parse_rss_feed(source_config["rss_url"], source_config)
            if articles:
                return articles[:limit]
            
            # Fallback to web scraping
            articles = await self._scrape_news_page(source_config)
            return articles[:limit]
            
        except Exception as e:
            print(f"Error getting news from {source}: {e}")
            return []
    
    async def _parse_rss_feed(self, rss_url: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse RSS feed for news articles."""
        try:
            response = await self.client.get(rss_url)
            response.raise_for_status()
            
            # Simple RSS parsing (in production, use feedparser library)
            content = response.text
            
            articles = []
            # This is a simplified RSS parser - in production, use proper XML parsing
            if "<item>" in content:
                # Extract basic article info from RSS
                articles.append({
                    "title": f"Latest news from {source_config['name']}",
                    "url": source_config["base_url"],
                    "source": source_config["name"],
                    "published_at": datetime.now().isoformat(),
                    "content_preview": f"Latest crypto news from {source_config['name']}",
                    "author": "Unknown"
                })
            
            return articles
            
        except Exception as e:
            print(f"Error parsing RSS feed {rss_url}: {e}")
            return []
    
    async def _scrape_news_page(self, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape news page for articles."""
        try:
            response = await self.client.get(source_config["base_url"])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            # Find article links (this is source-specific and simplified)
            article_links = soup.find_all('a', href=True)
            
            for link in article_links[:10]:  # Limit to first 10 links
                href = link.get('href')
                title = link.get_text(strip=True)
                
                if href and title and len(title) > 10:
                    # Make URL absolute
                    if href.startswith('/'):
                        href = urljoin(source_config["base_url"], href)
                    
                    articles.append({
                        "title": title,
                        "url": href,
                        "source": source_config["name"],
                        "published_at": datetime.now().isoformat(),
                        "content_preview": title,
                        "author": "Unknown"
                    })
            
            return articles
            
        except Exception as e:
            print(f"Error scraping news page {source_config['base_url']}: {e}")
            return []
    
    async def get_article_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get full article content from URL."""
        await self.initialize()
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "Unknown Title"
            
            # Extract content (simplified - in production, use more sophisticated extraction)
            content_divs = soup.find_all(['div', 'article', 'main'], class_=lambda x: x and any(
                keyword in x.lower() for keyword in ['content', 'article', 'body', 'main']
            ))
            
            content_text = ""
            if content_divs:
                content_text = content_divs[0].get_text(strip=True)
            else:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                content_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            # Limit content length
            if len(content_text) > 2000:
                content_text = content_text[:2000] + "..."
            
            return {
                "title": title_text,
                "url": url,
                "content": content_text,
                "scraped_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting article content from {url}: {e}")
            return None