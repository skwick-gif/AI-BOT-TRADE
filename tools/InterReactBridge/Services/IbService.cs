using InterReact;
using Microsoft.Extensions.Logging;
using System.Reactive.Linq;
using System.Reactive.Threading.Tasks;
using System.Text.Json;
using System.IO;

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
            _client = await InterReactClient.ConnectAsync(options =>
            {
                options.TwsIpAddress = System.Net.IPAddress.Parse(host);
                options.IBPortAddresses = new[] { port };
                options.TwsClientId = clientId;
            });

            // Try to get managed accounts with timeout
            try
            {
                var cts = new CancellationTokenSource(5000);
                var managedAccounts = await _client.Response.OfType<ManagedAccounts>().FirstAsync().ToTask(cts.Token);
                _accountCode = managedAccounts.Accounts.Split(',')[0];
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, Account: {Account}", host, port, _accountCode);
            }
            catch
            {
                _accountCode = null;
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, no account code received", host, port);
            }

            await WriteConnectionStatusAsync(new { connected = true, host, port, clientId, account = _accountCode, time = DateTime.UtcNow });
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to connect to IBKR");
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
            await Task.Delay(10000);

            sub.Dispose();

            // Cancel account summary request
            _client.Request.CancelAccountSummary(requestId);

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
            await Task.Delay(10000);

            sub.Dispose();

            // Cancel positions request
            _client.Request.CancelPositions();

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

    public async Task<object> PlaceOrderAsync(string symbol, string secType, string exchange, string action, int quantity, double? price = null, string? orderType = "LMT")
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

            var order = new Order()
            {
                Action = action.ToUpper() == "BUY" ? OrderAction.Buy : OrderAction.Sell,
                TotalQuantity = quantity,
                OrderType = orderType switch
                {
                    "MKT" => OrderTypes.Market,
                    "LMT" => OrderTypes.Limit,
                    _ => OrderTypes.Limit
                },
                LimitPrice = price ?? 0
            };

            var orderId = _client.Request.GetNextId();

            _client.Request.PlaceOrder(orderId, order, contract);

            _logger.LogInformation("Placed order: {OrderId} for {Symbol} {Action} {Quantity} @ {Price}", orderId, symbol, action, quantity, price);

            return new { OrderId = orderId, Status = "Placed" };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing order");
            throw;
        }
    }

    public async Task<object> GetScannerAsync(string scanType = "TOP_PERC_GAIN", int numberOfRows = 10)
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            var scanner = new ScannerSubscription()
            {
                ScanCode = scanType,
                NumberOfRows = numberOfRows
            };

            var results = new List<object>();

            var sub = _client.Response.OfType<ScannerData>()
                .Subscribe(data =>
                {
                    foreach (var item in data.Items)
                    {
                        results.Add(new
                        {
                            Rank = item.Rank,
                            ContractDetails = new
                            {
                                Symbol = item.ContractDetails.Contract.Symbol,
                                SecurityType = item.ContractDetails.Contract.SecurityType,
                                Exchange = item.ContractDetails.Contract.Exchange
                            },
                            Distance = item.Distance,
                            Benchmark = item.Benchmark,
                            Projection = item.Projection,
                            ComboLegs = item.ComboLegs
                        });
                    }
                });

            var reqId = _client.Request.GetNextId();
            _client.Request.RequestScannerSubscription(reqId, scanner);

            await Task.Delay(5000); // Wait for results

            sub.Dispose();
            _client.Request.CancelScannerSubscription(reqId);

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting scanner data");
            throw;
        }
    }

    public async Task<object> GetOptionsChainAsync(string underlyingSymbol, string exchange = "SMART")
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

        try
        {
            var underlying = new Contract()
            {
                Symbol = underlyingSymbol,
                SecurityType = "STK",
                Exchange = exchange,
                Currency = "USD"
            };

            var options = new List<object>();

            var sub = _client.Response.OfType<SecurityDefinitionOptionParameter>()
                .Subscribe(param =>
                {
                    options.Add(new
                    {
                        Exchange = param.Exchange,
                        UnderlyingContractId = param.UnderlyingContractId,
                        TradingClass = param.TradingClass,
                        Multiplier = param.Multiplier,
                        Expirations = param.Expirations,
                        Strikes = param.Strikes
                    });
                });

            var reqId = _client.Request.GetNextId();
            _client.Request.RequestSecurityDefinitionOptionalParameters(reqId, underlying.Symbol, "", underlying.SecurityType, underlying.ContractId);

            await Task.Delay(5000); // Wait for response

            sub.Dispose();

            return options;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting options chain");
            throw;
        }
    }

    public async Task GetLiveMarketData(HttpResponse response, string symbol, string secType, string exchange)
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-cache";
        response.Headers.Connection = "keep-alive";

        var contract = new Contract()
        {
            Symbol = symbol,
            SecurityType = secType,
            Exchange = exchange,
            Currency = "USD"
        };

        var sub = _client.Service
            .CreateMarketDataObservable(contract)
            .OfTickClass(selector => selector.PriceTick)
            .Subscribe(async pt =>
            {
                var data = new
                {
                    RequestId = pt.RequestId,
                    TickType = pt.TickType.ToString(),
                    Price = pt.Price,
                    Time = DateTime.UtcNow
                };
                var json = JsonSerializer.Serialize(data);
                await response.WriteAsync($"data: {json}\n\n");
                await response.Body.FlushAsync();
            });

        // Keep the connection open
        await Task.Delay(Timeout.Infinite, CancellationToken.None);
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