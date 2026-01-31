from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class NewsSource(Base):
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    base_url: Mapped[Optional[str]] = mapped_column(String(255))
    enabled: Mapped[bool] = mapped_column(default=True)
    crawl_depth: Mapped[int] = mapped_column(default=1)
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    articles: Mapped[list[NewsArticle]] = relationship("NewsArticle", back_populates="source")


class NewsArticle(Base):
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("newssource.id", ondelete="SET NULL"), index=True)
    url: Mapped[str] = mapped_column(String(1024))
    canonical_url: Mapped[Optional[str]] = mapped_column(String(1024))
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    title: Mapped[Optional[str]] = mapped_column(String(512))
    author: Mapped[Optional[str]] = mapped_column(String(256))
    language: Mapped[Optional[str]] = mapped_column(String(16))
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[Optional[str]] = mapped_column(String(32))
    content_text: Mapped[Optional[str]] = mapped_column(Text)
    content_html: Mapped[Optional[str]] = mapped_column(Text)

    source: Mapped[Optional[NewsSource]] = relationship("NewsSource", back_populates="articles")
    sentiments: Mapped[list[NewsSentiment]] = relationship("NewsSentiment", back_populates="article")
    topics: Mapped[Optional[NewsTopics]] = relationship("NewsTopics", back_populates="article", uselist=False)


class NewsSentiment(Base):
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("newsarticle.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(32))
    score: Mapped[float]
    confidence: Mapped[Optional[float]]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    article: Mapped[NewsArticle] = relationship("NewsArticle", back_populates="sentiments")


class NewsTopics(Base):
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("newsarticle.id", ondelete="CASCADE"), index=True)
    labels: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    article: Mapped[NewsArticle] = relationship("NewsArticle", back_populates="topics")


