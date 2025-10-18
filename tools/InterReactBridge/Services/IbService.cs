using InterReact;
using Microsoft.Extensions.Logging;
using System.Reactive.Linq;
using System.Reactive.Threading.Tasks;
using System.Text.Json;
using System.IO;
using InterReactBridge.Models;

namespace InterReactBridge.Services;

public class IbService
{
    private readonly ILogger<IbService> _logger;
    private readonly TwsConnectionService _twsConnection;
    private IInterReactClient? _client;
    private string? _accountCode;

    public IbService(ILogger<IbService> logger, TwsConnectionService twsConnection)
    {
        _logger = logger;
        _twsConnection = twsConnection;
    }

    public async Task<bool> ConnectAsync(string host, int port, int clientId)
    {
        try
        {
            _logger.LogInformation("=== Starting IBKR connection attempt ===");
            _logger.LogInformation("Host: {Host}, Port: {Port}, ClientId: {ClientId}", host, port, clientId);
            
            _logger.LogInformation("Creating InterReactClient...");
            _client = await InterReactClient.ConnectAsync(options =>
            {
                options.TwsIpAddress = System.Net.IPAddress.Parse(host);
                options.IBPortAddresses = new[] { port };
                options.TwsClientId = clientId;
                _logger.LogInformation("Options set: IP={IP}, Port={Port}, ClientId={ClientId}", 
                    options.TwsIpAddress, port, clientId);
            });
            
            _logger.LogInformation("InterReactClient created successfully");

            _logger.LogInformation("InterReactClient created successfully");

            // Try to get managed accounts with timeout
            try
            {
                _logger.LogInformation("Waiting for ManagedAccounts response...");
                var cts = new CancellationTokenSource(5000);
                var managedAccounts = await _client.Response.OfType<ManagedAccounts>().FirstAsync().ToTask(cts.Token);
                _accountCode = managedAccounts.Accounts.Split(',')[0];
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, Account: {Account}", host, port, _accountCode);
            }
            catch (Exception accountEx)
            {
                _accountCode = null;
                _logger.LogWarning(accountEx, "Could not retrieve account code within timeout");
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, no account code received", host, port);
            }

            await WriteConnectionStatusAsync(new { connected = true, host, port, clientId, account = _accountCode, time = DateTime.UtcNow });
            _logger.LogInformation("=== Connection successful ===");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "=== Failed to connect to IBKR ===");
            _logger.LogError("Error Type: {Type}", ex.GetType().Name);
            _logger.LogError("Error Message: {Message}", ex.Message);
            if (ex.InnerException != null)
            {
                _logger.LogError("Inner Exception: {InnerMessage}", ex.InnerException.Message);
            }
            await WriteConnectionStatusAsync(new { connected = false, host, port, clientId, error = ex.Message, time = DateTime.UtcNow });
            return false;
        }
    }

    public async Task<object> GetAccountSummary()
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting account summary using RequestAccountSummary...");
            
            // Use direct RequestAccountSummary approach
            var summaries = new List<AccountSummary>();
            
            var sub = client.Response.OfType<AccountSummary>()
                .Subscribe(summary => 
                {
                    _logger.LogInformation("Received AccountSummary: Account={Account}, Tag={Tag}, Value={Value}, Currency={Currency}", 
                        summary.Account, summary.Tag, summary.Value, summary.Currency);
                    summaries.Add(summary);
                });

            // Request account summary directly with specific request ID
            var requestId = client.Request.GetNextId();
            client.Request.RequestAccountSummary(requestId, "All");

            // Wait for summaries
            await Task.Delay(5000);

            sub.Dispose();

            // Note: Not canceling subscription to avoid connection issues
            // TWS will stop sending when we unsubscribe

            _logger.LogInformation("Received {Count} account summary items", summaries.Count);
            // persist a small audit file with count
            await WriteConnectionStatusAsync(new { accountSummaryCount = summaries.Count, time = DateTime.UtcNow });

            // Convert to dictionary grouped by Tag for UI compatibility
            // UI expects: { "NetLiquidation": { value: "123", currency: "USD", account: "U123" } }
            var dictionary = new Dictionary<string, object>();
            foreach (var item in summaries)
            {
                var tag = string.IsNullOrEmpty(item.Tag) ? "Unknown" : item.Tag;
                
                // If tag already exists (multiple accounts), keep the first one
                // TODO: In future, support multiple accounts by returning array per tag
                if (!dictionary.ContainsKey(tag))
                {
                    dictionary[tag] = new
                    {
                        value = string.IsNullOrEmpty(item.Value) ? "0" : item.Value,
                        currency = string.IsNullOrEmpty(item.Currency) ? "USD" : item.Currency,
                        account = string.IsNullOrEmpty(item.Account) ? "Unknown" : item.Account
                    };
                }
            }
            
            _logger.LogInformation("Converted to dictionary with {Count} unique tags", dictionary.Count);
            return dictionary;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting account summary");
            throw;
        }
    }

    public async Task<object> GetPortfolio()
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting portfolio positions using RequestPositions...");
            
            // Use direct RequestPositions approach
            var positions = new List<AccountPosition>();
            
            var sub = client.Response.OfType<AccountPosition>()
                .Subscribe(position => 
                {
                    _logger.LogInformation("Received AccountPosition: Account={Account}, Symbol={Symbol}, Position={Position}, AverageCost={AverageCost}", 
                        position.Account, position.Contract.Symbol, position.Position, position.AverageCost);
                    positions.Add(position);
                });

            // Request positions directly
            client.Request.RequestPositions();

            // Wait for positions
            await Task.Delay(5000);

            sub.Dispose();

            // Note: Not canceling subscription to avoid connection issues
            // TWS will stop sending when we unsubscribe

            _logger.LogInformation("Received {Count} portfolio positions", positions.Count);
            // persist a small audit file with count
            await WriteConnectionStatusAsync(new { portfolioCount = positions.Count, time = DateTime.UtcNow });

            // Normalize field names to snake_case for Python UI compatibility
            // UI expects: { symbol, position, average_cost, market_price, market_value, unrealized_pnl }
            return positions.Select(p => new
            {
                account = string.IsNullOrEmpty(p.Account) ? "Unknown" : p.Account,
                symbol = string.IsNullOrEmpty(p.Contract.Symbol) ? "Unknown" : p.Contract.Symbol,
                security_type = string.IsNullOrEmpty(p.Contract.SecurityType) ? "Unknown" : p.Contract.SecurityType,
                exchange = string.IsNullOrEmpty(p.Contract.Exchange) ? "Unknown" : p.Contract.Exchange,
                currency = string.IsNullOrEmpty(p.Contract.Currency) ? "USD" : p.Contract.Currency,
                position = p.Position,
                average_cost = p.AverageCost,
                market_price = p.AverageCost,  // TODO: Fetch real-time market price from TWS
                market_value = p.Position * (decimal)p.AverageCost,
                unrealized_pnl = 0.0  // TODO: Calculate from (market_price - average_cost) * position
            }).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting portfolio");
            throw;
        }
    }

    public async Task<object> GetMarketData(string symbol, string secType, string exchange, TimeSpan duration)
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            var contract = new Contract()
            {
                Symbol = symbol,
                SecurityType = secType,
                Exchange = exchange,
                Currency = "USD"
            };

            var ticks = new List<object>();

            var sub = client.Service
                .CreateMarketDataObservable(contract)
                .OfTickClass(selector => selector.PriceTick)
                .Subscribe(pt =>
                {
                    // collect basic tick info
                    ticks.Add(new {
                        RequestId = pt.RequestId,
                        TickType = pt.TickType.ToString(),
                        Price = pt.Price,
                        Time = DateTime.UtcNow
                    });
                });

            await Task.Delay(duration);

            sub.Dispose();

            return ticks;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting market data");
            throw;
        }
    }

    /// <summary>
    /// Get delayed market prices for calculating indicators
    /// IBKR provides delayed data (15-20 minutes delay for free accounts)
    /// We'll collect price ticks over time to build a price series
    /// </summary>
    /// <param name="symbol">Stock symbol (e.g., "AAPL")</param>
    /// <param name="secType">Security type (e.g., "STK" for stock)</param>
    /// <param name="exchange">Exchange (e.g., "SMART")</param>
    /// <param name="sampleSeconds">How many seconds to collect ticks (default 30)</param>
    /// <returns>List of price points sampled over time</returns>
    public async Task<PriceSeriesResponse> GetDelayedPriceSeries(
        string symbol, 
        string secType, 
        string exchange, 
        int sampleSeconds = 30)
    {
        var client = _twsConnection.GetClient();
        if (client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting delayed price series: {Symbol}, Duration={Duration}s", 
                symbol, sampleSeconds);

            var contract = new Contract()
            {
                Symbol = symbol,
                SecurityType = secType,
                Exchange = exchange,
                Currency = "USD"
            };

            var prices = new List<PricePoint>();
            var lastPrice = 0.0;

            var sub = client.Service
                .CreateMarketDataObservable(contract)
                .OfTickClass(selector => selector.PriceTick)
                .Subscribe(pt =>
                {
                    // Collect price ticks
                    if (pt.Price > 0 && pt.Price != lastPrice)
                    {
                        lastPrice = pt.Price;
                        prices.Add(new PricePoint
                        {
                            Price = pt.Price,
                            TickType = pt.TickType.ToString(),
                            Time = DateTime.UtcNow
                        });
                        _logger.LogDebug("Price tick: {Price} at {Time}", pt.Price, DateTime.UtcNow);
                    }
                });

            // Collect ticks for the specified duration
            await Task.Delay(TimeSpan.FromSeconds(sampleSeconds));

            sub.Dispose();

            _logger.LogInformation("Collected {Count} price points for {Symbol}", prices.Count, symbol);

            return new PriceSeriesResponse
            {
                Symbol = symbol,
                SampleSeconds = sampleSeconds,
                PricesCount = prices.Count,
                Prices = prices,
                Note = "Delayed market data from IBKR (15-20 minutes delay for free accounts). Limited to ticks received during sample period."
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting delayed price series for {Symbol}", symbol);
            throw;
        }
    }

    private async Task WriteConnectionStatusAsync(object obj)
    {
        try
        {
            // prefer the current working directory (content root) so the status file is adjacent to the app files
            var folder = Directory.GetCurrentDirectory();
            var path = Path.Combine(folder, "interreact_status.json");
            var opts = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(obj, opts);
            await File.WriteAllTextAsync(path, json);
        }
        catch
        {
            // ignore write failures to avoid throwing during normal operation
        }
    }
}
