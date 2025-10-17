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
    private IInterReactClient? _client;
    private string? _accountCode;

    public IbService(ILogger<IbService> logger)
    {
        _logger = logger;
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
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting account summary using RequestAccountSummary...");
            
            // Use direct RequestAccountSummary approach
            var summaries = new List<AccountSummary>();
            
            var sub = _client.Response.OfType<AccountSummary>()
                .Subscribe(summary => 
                {
                    _logger.LogInformation("Received AccountSummary: Account={Account}, Tag={Tag}, Value={Value}, Currency={Currency}", 
                        summary.Account, summary.Tag, summary.Value, summary.Currency);
                    summaries.Add(summary);
                });

            // Request account summary directly with specific request ID
            var requestId = _client.Request.GetNextId();
            _client.Request.RequestAccountSummary(requestId, "All");

            // Wait for summaries
            await Task.Delay(5000);

            sub.Dispose();

            // Note: Not canceling subscription to avoid connection issues
            // TWS will stop sending when we unsubscribe

            _logger.LogInformation("Received {Count} account summary items", summaries.Count);
            // persist a small audit file with count
            await WriteConnectionStatusAsync(new { accountSummaryCount = summaries.Count, time = DateTime.UtcNow });

            return summaries.Select(x => new 
            { 
                Tag = string.IsNullOrEmpty(x.Tag) ? "Unknown" : x.Tag,
                Value = string.IsNullOrEmpty(x.Value) ? "N/A" : x.Value,
                Account = string.IsNullOrEmpty(x.Account) ? "Unknown" : x.Account,
                Currency = string.IsNullOrEmpty(x.Currency) ? "USD" : x.Currency
            }).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting account summary");
            throw;
        }
    }

    public async Task<object> GetPortfolio()
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            _logger.LogInformation("Requesting portfolio positions using RequestPositions...");
            
            // Use direct RequestPositions approach
            var positions = new List<AccountPosition>();
            
            var sub = _client.Response.OfType<AccountPosition>()
                .Subscribe(position => 
                {
                    _logger.LogInformation("Received AccountPosition: Account={Account}, Symbol={Symbol}, Position={Position}, AverageCost={AverageCost}", 
                        position.Account, position.Contract.Symbol, position.Position, position.AverageCost);
                    positions.Add(position);
                });

            // Request positions directly
            _client.Request.RequestPositions();

            // Wait for positions
            await Task.Delay(5000);

            sub.Dispose();

            // Note: Not canceling subscription to avoid connection issues
            // TWS will stop sending when we unsubscribe

            _logger.LogInformation("Received {Count} portfolio positions", positions.Count);
            // persist a small audit file with count
            await WriteConnectionStatusAsync(new { portfolioCount = positions.Count, time = DateTime.UtcNow });

            return positions.Select(p => new
            {
                Account = string.IsNullOrEmpty(p.Account) ? "Unknown" : p.Account,
                Symbol = string.IsNullOrEmpty(p.Contract.Symbol) ? "Unknown" : p.Contract.Symbol,
                SecurityType = string.IsNullOrEmpty(p.Contract.SecurityType) ? "Unknown" : p.Contract.SecurityType,
                Exchange = string.IsNullOrEmpty(p.Contract.Exchange) ? "Unknown" : p.Contract.Exchange,
                Currency = string.IsNullOrEmpty(p.Contract.Currency) ? "USD" : p.Contract.Currency,
                Position = p.Position,
                AverageCost = p.AverageCost,
                MarketValue = p.Position * (decimal)p.AverageCost
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
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

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

            var sub = _client.Service
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
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

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

            var sub = _client.Service
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
