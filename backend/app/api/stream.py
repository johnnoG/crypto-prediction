from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

try:
    from services.prices_service import get_simple_price_with_cache, get_market_data_with_cache
    from cache import AsyncCache
except ImportError:
    from services.prices_service import get_simple_price_with_cache, get_market_data_with_cache
    from cache import AsyncCache

router = APIRouter(prefix="/stream", tags=["streaming"])


class ConnectionManager:
    """Manages WebSocket connections for real-time price streaming."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Remove dead connections
                self.active_connections.remove(connection)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time price streaming.
    
    Sends periodic price updates for major cryptocurrencies.
    """
    try:
        await manager.connect(websocket)
        print(f"WebSocket client connected. Total connections: {len(manager.active_connections)}")
        
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "message": "WebSocket connected successfully",
                "timestamp": asyncio.get_event_loop().time()
            }),
            websocket
        )
        
        # Send initial price data
        try:
            initial_data = await get_simple_price_with_cache(
                ids="bitcoin,ethereum,binancecoin,cardano,solana,polkadot,chainlink,uniswap,avalanche-2,matic-network",
                vs_currencies="usd"
            )
            
            await manager.send_personal_message(
                json.dumps({
                    "type": "initial_prices",
                    "data": initial_data,
                    "timestamp": asyncio.get_event_loop().time()
                }),
                websocket
            )
        except Exception as e:
            print(f"Failed to send initial data: {e}")
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(15)  # Update every 15 seconds for better real-time feel
            
            try:
                # Get fresh price data
                price_data = await get_simple_price_with_cache(
                    ids="bitcoin,ethereum,binancecoin,cardano,solana,polkadot,chainlink,uniswap,avalanche-2,matic-network",
                    vs_currencies="usd"
                )
                
                market_data = await get_market_data_with_cache(
                    ids="bitcoin,ethereum,binancecoin,cardano,solana,polkadot,chainlink,uniswap,avalanche-2,matic-network",
                    vs_currency="usd"
                )
                
                # Combine price and market data
                combined_data = {}
                for crypto_id in price_data.keys():
                    combined_data[crypto_id] = {
                        "price": price_data[crypto_id].get("usd", 0),
                        "change_24h": market_data.get(crypto_id, {}).get("price_change_24h", 0),
                        "market_cap": market_data.get(crypto_id, {}).get("market_cap", 0),
                        "volume_24h": market_data.get(crypto_id, {}).get("volume_24h", 0),
                    }
                
                message = json.dumps({
                    "type": "price_update",
                    "data": combined_data,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                await manager.send_personal_message(message, websocket)
                
            except Exception as e:
                print(f"Error in WebSocket update loop: {e}")
                # Send error message but keep connection alive
                error_message = json.dumps({
                    "type": "error",
                    "message": f"Failed to fetch price data: {str(e)}",
                    "timestamp": asyncio.get_event_loop().time()
                })
                try:
                    await manager.send_personal_message(error_message, websocket)
                except:
                    break  # Connection is dead, exit loop
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/sse")
async def sse_endpoint():
    """Server-Sent Events endpoint as fallback for WebSocket streaming.
    
    Provides real-time price updates via SSE for clients that can't use WebSocket.
    """
    
    async def generate_price_stream():
        cache = AsyncCache()
        await cache.initialize()
        
        while True:
            try:
                # Get current price data
                price_data = await get_simple_price_with_cache(
                    ids="bitcoin,ethereum,binancecoin,cardano,solana,polkadot",
                    vs_currencies="usd"
                )
                
                market_data = await get_market_data_with_cache(
                    ids="bitcoin,ethereum,binancecoin,cardano,solana,polkadot",
                    vs_currency="usd"
                )
                
                # Combine data
                combined_data = {}
                for crypto_id in price_data.keys():
                    combined_data[crypto_id] = {
                        "price": price_data[crypto_id].get("usd", 0),
                        "change_24h": market_data.get(crypto_id, {}).get("price_change_24h", 0),
                        "market_cap": market_data.get(crypto_id, {}).get("market_cap", 0),
                        "volume_24h": market_data.get(crypto_id, {}).get("volume_24h", 0),
                    }
                
                # Format as SSE
                event_data = json.dumps({
                    "type": "price_update",
                    "data": combined_data,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                yield f"data: {event_data}\n\n"
                
                # Wait 30 seconds before next update
                await asyncio.sleep(30)
                
            except Exception as e:
                # Send error event
                error_data = json.dumps({
                    "type": "error",
                    "message": f"Failed to fetch data: {str(e)}",
                    "timestamp": asyncio.get_event_loop().time()
                })
                yield f"data: {error_data}\n\n"
                await asyncio.sleep(60)  # Wait longer on error
    
    return StreamingResponse(
        generate_price_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )


@router.get("/snapshot")
async def get_streaming_snapshot() -> Dict[str, Any]:
    """Get a snapshot of current streaming data for fallback scenarios."""
    
    try:
        # Return major cryptos that the frontend expects
        major_ids = "bitcoin,ethereum,tether,binancecoin,solana,ripple,usd-coin,cardano,avalanche-2,dogecoin,polkadot,matic-network,chainlink,litecoin,uniswap,bitcoin-cash,cosmos,stellar,ethereum-classic,filecoin,monero"
        
        price_data = await get_simple_price_with_cache(
            ids=major_ids,
            vs_currencies="usd"
        )
        
        market_data = await get_market_data_with_cache(
            ids=major_ids,
            vs_currency="usd"
        )
        
        # Combine data
        combined_data = {}
        for crypto_id in price_data.keys():
            crypto_market = market_data.get(crypto_id, {})
            price = price_data[crypto_id].get("usd", 0)
            
            # Only include cryptos with valid prices
            if price and price > 0:
                combined_data[crypto_id] = {
                    "price": price,
                    "change_24h": crypto_market.get("price_change_24h") or crypto_market.get("price_change_percentage_24h") or None,
                    "market_cap": crypto_market.get("market_cap", 0),
                    "volume_24h": crypto_market.get("volume_24h", 0),
                }
        
        print(f"[INFO] Stream snapshot: Returning data for {len(combined_data)} cryptocurrencies")
        
        return {
            "data": combined_data,
            "timestamp": asyncio.get_event_loop().time(),
            "type": "snapshot",
            "active_connections": len(manager.active_connections),
        }
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        print(f"[ERROR] Stream snapshot failed: {error_detail}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        
        # Always return proper structure with empty data on error
        # This ensures frontend can render UI even if data fetch fails
        return {
            "data": {},
            "timestamp": asyncio.get_event_loop().time(),
            "type": "error_snapshot",
            "error": error_detail,
            "active_connections": len(manager.active_connections),
        }

