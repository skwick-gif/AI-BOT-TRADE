using Microsoft.AspNetCore.SignalR;

namespace InterReactBridge.Hubs;

/// <summary>
/// SignalR Hub for real-time market data streaming
/// Allows clients to subscribe/unsubscribe to specific symbols
/// </summary>
public class MarketDataHub : Hub
{
    private readonly ILogger<MarketDataHub> _logger;
    private static readonly Dictionary<string, HashSet<string>> SymbolSubscriptions = new();

    public MarketDataHub(ILogger<MarketDataHub> logger)
    {
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        _logger.LogInformation("Client connected to MarketDataHub: {ConnectionId}", Context.ConnectionId);
        await base.OnConnectedAsync();
        
        await Clients.Caller.SendAsync("Connected", new { connectionId = Context.ConnectionId });
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _logger.LogInformation("Client disconnected from MarketDataHub: {ConnectionId}", Context.ConnectionId);
        
        // Clean up subscriptions
        lock (SymbolSubscriptions)
        {
            foreach (var symbol in SymbolSubscriptions.Keys.ToList())
            {
                SymbolSubscriptions[symbol].Remove(Context.ConnectionId);
                if (SymbolSubscriptions[symbol].Count == 0)
                {
                    SymbolSubscriptions.Remove(symbol);
                    _logger.LogInformation("No more subscribers for {Symbol}, removing", symbol);
                }
            }
        }
        
        await base.OnDisconnectedAsync(exception);
    }

    /// <summary>
    /// Subscribe to real-time market data for a symbol
    /// </summary>
    /// <param name="symbol">Stock symbol (e.g., "AAPL")</param>
    /// <param name="secType">Security type (e.g., "STK")</param>
    /// <param name="exchange">Exchange (e.g., "SMART")</param>
    public async Task SubscribeToSymbol(string symbol, string secType = "STK", string exchange = "SMART")
    {
        _logger.LogInformation("Client {ConnectionId} subscribing to {Symbol}", Context.ConnectionId, symbol);
        
        lock (SymbolSubscriptions)
        {
            if (!SymbolSubscriptions.ContainsKey(symbol))
            {
                SymbolSubscriptions[symbol] = new HashSet<string>();
            }
            SymbolSubscriptions[symbol].Add(Context.ConnectionId);
        }
        
        await Clients.Caller.SendAsync("SubscribedToSymbol", new 
        { 
            symbol, 
            secType, 
            exchange, 
            subscriberCount = SymbolSubscriptions[symbol].Count 
        });
        
        // Background service will start streaming data
    }

    /// <summary>
    /// Unsubscribe from market data for a symbol
    /// </summary>
    public async Task UnsubscribeFromSymbol(string symbol)
    {
        _logger.LogInformation("Client {ConnectionId} unsubscribing from {Symbol}", Context.ConnectionId, symbol);
        
        lock (SymbolSubscriptions)
        {
            if (SymbolSubscriptions.ContainsKey(symbol))
            {
                SymbolSubscriptions[symbol].Remove(Context.ConnectionId);
                if (SymbolSubscriptions[symbol].Count == 0)
                {
                    SymbolSubscriptions.Remove(symbol);
                }
            }
        }
        
        await Clients.Caller.SendAsync("UnsubscribedFromSymbol", new { symbol });
    }

    /// <summary>
    /// Get list of currently subscribed symbols
    /// </summary>
    public async Task GetSubscribedSymbols()
    {
        var symbols = SymbolSubscriptions.Keys.ToList();
        await Clients.Caller.SendAsync("SubscribedSymbols", new { symbols });
    }

    /// <summary>
    /// Helper method to check if a symbol has any subscribers
    /// </summary>
    public static bool HasSubscribers(string symbol)
    {
        lock (SymbolSubscriptions)
        {
            return SymbolSubscriptions.ContainsKey(symbol) && SymbolSubscriptions[symbol].Count > 0;
        }
    }

    /// <summary>
    /// Helper method to get connection IDs subscribed to a symbol
    /// </summary>
    public static IEnumerable<string> GetSubscribers(string symbol)
    {
        lock (SymbolSubscriptions)
        {
            return SymbolSubscriptions.ContainsKey(symbol) 
                ? SymbolSubscriptions[symbol].ToList() 
                : Enumerable.Empty<string>();
        }
    }
}
