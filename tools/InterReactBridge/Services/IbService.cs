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
    private string? _lastError;
    private string? _lastErrorCode;
    private bool _connected;
    private DateTime? _lastAttemptUtc;
    private string? _lastHost;
    private int _lastPort;
    private int _lastClientId;

    public string? LastError => _lastError;
    public bool IsConnected => _connected;
    public object GetStatus() => new
    {
        connected = _connected,
        errorCode = _lastErrorCode,
        errorMessage = _lastError,
        hint = GetHintFromErrorCode(_lastErrorCode),
        account = _accountCode,
        lastAttemptUtc = _lastAttemptUtc,
        host = _lastHost,
        port = _lastPort,
        clientId = _lastClientId
    };

    public IbService(ILogger<IbService> logger)
    {
        _logger = logger;
    }

    public async Task<bool> ConnectAsync(string host, int port, int clientId)
    {
        try
        {
            _logger.LogInformation("/connect requested: {Host}:{Port} clientId={ClientId}", host, port, clientId);
            _lastHost = host; _lastPort = port; _lastClientId = clientId; _lastAttemptUtc = DateTime.UtcNow; _connected = false; _lastError = null; _lastErrorCode = null;

            // Preflight: ensure TCP port is reachable before attempting InterReact connect
            try
            {
                using var tcp = new System.Net.Sockets.TcpClient();
                var preflightCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(800));
                var preflightTask = tcp.ConnectAsync(host, port);
                var done = await Task.WhenAny(preflightTask, Task.Delay(Timeout.Infinite, preflightCts.Token));
                if (done != preflightTask || !tcp.Connected)
                {
                    _logger.LogWarning("TCP preflight failed to {Host}:{Port}", host, port);
                    _lastErrorCode = "TCP_PREFLIGHT_FAILED";
                    _lastError = "tcp preflight failed";
                    await PersistStatusAsync();
                    return false;
                }
            }
            catch (Exception pex)
            {
                _logger.LogWarning(pex, "TCP preflight exception to {Host}:{Port}", host, port);
                _lastErrorCode = "TCP_PREFLIGHT_EXCEPTION";
                _lastError = "tcp preflight exception";
                await PersistStatusAsync();
                return false;
            }

            // Enforce a connection timeout so HTTP request won't hang indefinitely
            var connectCts = new CancellationTokenSource(TimeSpan.FromSeconds(20));

            var connectTask = InterReactClient.ConnectAsync(options =>
            {
                options.TwsIpAddress = System.Net.IPAddress.Parse(host);
                options.IBPortAddresses = new[] { port };
                options.TwsClientId = clientId;
            });

            var completed = await Task.WhenAny(connectTask, Task.Delay(Timeout.Infinite, connectCts.Token));
            if (completed != connectTask)
            {
                _logger.LogWarning("ConnectAsync timed out after 20s to {Host}:{Port}", host, port);
                _lastErrorCode = "CONNECT_TIMEOUT";
                _lastError = "connect timeout (20s)";
                await PersistStatusAsync();
                return false;
            }
            _client = await connectTask; // propagate exception if any

            // Try to get managed accounts with timeout
            try
            {
                var cts = new CancellationTokenSource(3000);
                var managedAccounts = await _client.Response.OfType<ManagedAccounts>().FirstAsync().ToTask(cts.Token);
                _accountCode = managedAccounts.Accounts.Split(',')[0];
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, Account: {Account}", host, port, _accountCode);
                _connected = true;
                _lastErrorCode = null;
                _lastError = null;
            }
            catch
            {
                _accountCode = null;
                _logger.LogInformation("Connected to IBKR at {Host}:{Port}, no account code received", host, port);
                _connected = true; // connected but account not yet received
                _lastErrorCode = null;
                _lastError = null;
            }
            await PersistStatusAsync();
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to connect to IBKR");
            _lastErrorCode = "CONNECT_EXCEPTION";
            _lastError = ex.Message;
            await PersistStatusAsync();
            return false;
        }
    }

    public (string? host, int port, int clientId) GetLastEndpoint() => (_lastHost, _lastPort, _lastClientId);

    public async Task<object> ProbeTcpAsync(string host, int port, int timeoutMs = 800)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            using var tcp = new System.Net.Sockets.TcpClient();
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(Math.Max(100, timeoutMs)));
            var connectTask = tcp.ConnectAsync(host, port);
            var done = await Task.WhenAny(connectTask, Task.Delay(Timeout.Infinite, cts.Token));
            sw.Stop();
            if (done == connectTask && tcp.Connected)
            {
                return new { reachable = true, latencyMs = sw.ElapsedMilliseconds };
            }
            return new { reachable = false, latencyMs = sw.ElapsedMilliseconds, errorCode = "TCP_PREFLIGHT_FAILED", errorMessage = "tcp preflight failed" };
        }
        catch (Exception ex)
        {
            sw.Stop();
            return new { reachable = false, latencyMs = sw.ElapsedMilliseconds, errorCode = "TCP_PREFLIGHT_EXCEPTION", errorMessage = ex.Message };
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

    private Task PersistStatusAsync()
    {
        // persist the standardized status payload
        return WriteConnectionStatusAsync(GetStatus());
    }

    private static string? GetHintFromErrorCode(string? code)
    {
        return code switch
        {
            "TCP_PREFLIGHT_FAILED" => "Port closed or blocked. Ensure IB Gateway/TWS is running and listening on this port.",
            "CONNECT_TIMEOUT" => "Likely API disabled or untrusted IP. In TWS/IB Gateway: enable 'Enable ActiveX and Socket Clients' and add 127.0.0.1 to Trusted IPs.",
            "TCP_PREFLIGHT_EXCEPTION" => "Host unreachable. Verify host and network; use 127.0.0.1 when running locally.",
            "CONNECT_EXCEPTION" => "Connection error. Check IB Gateway/TWS API settings and logs.",
            _ => null
        };
    }

    // -----------------------------
    // Advanced order helpers
    // -----------------------------

    public async Task<object> PlaceBracketOrderAsync(InterReactBridge.Models.BracketOrderRequest req)
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");

        var contract = new Contract()
        {
            Symbol = req.Symbol,
            SecurityType = req.SecType,
            Exchange = req.Exchange,
            Currency = "USD"
        };

        // Parent (entry)
        var entryOrder = new Order()
        {
            Action = req.Action.ToUpper() == "BUY" ? OrderAction.Buy : OrderAction.Sell,
            TotalQuantity = req.Quantity,
            OrderType = (req.EntryType?.ToUpper()) switch
            {
                "MKT" => OrderTypes.Market,
                _ => OrderTypes.Limit
            },
            LimitPrice = (req.EntryType?.ToUpper() == "LMT") ? (req.EntryPrice ?? 0) : 0,
            OutsideRth = req.OutsideRth,
            Transmit = false
        };
        var parentId = _client.Request.GetNextId();
        _client.Request.PlaceOrder(parentId, entryOrder, contract);

        // Take profit child
        int tpId = -1;
        if (req.TakeProfitPrice.HasValue)
        {
            var tp = new Order()
            {
                Action = entryOrder.Action == OrderAction.Buy ? OrderAction.Sell : OrderAction.Buy,
                TotalQuantity = req.Quantity,
                OrderType = OrderTypes.Limit,
                LimitPrice = req.TakeProfitPrice.Value,
                ParentId = parentId,
                Transmit = false,
                OutsideRth = req.OutsideRth
            };
            tpId = _client.Request.GetNextId();
            _client.Request.PlaceOrder(tpId, tp, contract);
        }

        // Stop loss child
        int slId = -1;
        if (req.StopLossPrice.HasValue)
        {
            var sl = new Order()
            {
                Action = entryOrder.Action == OrderAction.Buy ? OrderAction.Sell : OrderAction.Buy,
                TotalQuantity = req.Quantity,
                OrderType = OrderTypes.Stop,
                AuxPrice = req.StopLossPrice.Value,
                ParentId = parentId,
                Transmit = true, // transmit the whole bracket with the last child
                OutsideRth = req.OutsideRth
            };
            slId = _client.Request.GetNextId();
            _client.Request.PlaceOrder(slId, sl, contract);
        }

        _logger.LogInformation("Placed bracket order parent={Parent} tp={Tp} sl={Sl} for {Symbol}", parentId, tpId, slId, req.Symbol);
        return new { success = true, parentOrderId = parentId, takeProfitOrderId = tpId, stopLossOrderId = slId };
    }

    public async Task<object> PlaceOcoOrdersAsync(InterReactBridge.Models.OcoOrderRequest req)
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");
        if (req.Orders == null || req.Orders.Count < 2) throw new ArgumentException("At least two orders required for OCO");

        var contract = new Contract()
        {
            Symbol = req.Symbol,
            SecurityType = req.SecType,
            Exchange = req.Exchange,
            Currency = "USD"
        };

        var group = string.IsNullOrWhiteSpace(req.OcaGroup) ? $"{req.Symbol}-OCO-{DateTime.UtcNow:yyyyMMddHHmmss}" : req.OcaGroup;
        var ids = new List<int>();

        foreach (var item in req.Orders)
        {
            var order = new Order()
            {
                Action = item.Action.ToUpper() == "BUY" ? OrderAction.Buy : OrderAction.Sell,
                TotalQuantity = item.Quantity,
                OrderType = (item.OrderType?.ToUpper()) switch
                {
                    "MKT" => OrderTypes.Market,
                    "STP" => OrderTypes.Stop,
                    _ => OrderTypes.Limit
                },
                LimitPrice = (item.OrderType?.ToUpper() == "LMT") ? (item.Price ?? 0) : 0,
                AuxPrice = (item.OrderType?.ToUpper() == "STP") ? (item.StopPrice ?? 0) : 0,
                OcaGroup = group,
                OcaType = 1,
                Transmit = true
            };
            var id = _client.Request.GetNextId();
            _client.Request.PlaceOrder(id, order, contract);
            ids.Add(id);
        }

        _logger.LogInformation("Placed OCO group {Group} with {Count} orders for {Symbol}", group, ids.Count, req.Symbol);
        return new { success = true, ocaGroup = group, orderIds = ids };
    }

    public async Task<object> PlaceComboOrderAsync(InterReactBridge.Models.ComboOrderRequest req)
    {
        if (_client == null) throw new InvalidOperationException("Not connected to IBKR.");
        if (req.Legs == null || req.Legs.Count == 0) throw new ArgumentException("At least one combo leg required");

        var bag = new Contract()
        {
            SecurityType = "BAG",
            Exchange = req.Exchange,
            Currency = req.Currency,
            ComboLegs = req.Legs.Select(l => new ComboLeg
            {
                ConId = l.ConId,
                Ratio = l.Ratio,
                Action = l.Action.ToUpper() == "BUY" ? ComboAction.Buy : ComboAction.Sell,
                Exchange = l.Exchange
            }).ToList()
        };

        var order = new Order()
        {
            TotalQuantity = req.Quantity,
            OrderType = (req.OrderType?.ToUpper()) switch
            {
                "MKT" => OrderTypes.Market,
                _ => OrderTypes.Limit
            },
            LimitPrice = (req.OrderType?.ToUpper() == "LMT") ? (req.Price ?? 0) : 0,
            Transmit = true
        };

        var id = _client.Request.GetNextId();
        _client.Request.PlaceOrder(id, order, bag);
        _logger.LogInformation("Placed combo order {Id} with {Legs} legs on {Exchange}", id, req.Legs.Count, req.Exchange);
        return new { success = true, orderId = id };
    }
}