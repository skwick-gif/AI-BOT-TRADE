using Grpc.Core;
using IBKRAdapterServer.Protos;
using System.Threading.Channels;
using IBApi;

namespace ExternalConnector;

public class ReflectionIbkrProvider : IProvider
{
    private EClientSocket _clientSocket;
    private IbkrWrapper _wrapper;
    private EReaderSignal? _signal;
    private EReader? _reader;
    private int _nextOrderId = 1;
    private readonly Dictionary<string, double> _accountData = new();
    private string _accountType = "Unknown";
    private readonly Dictionary<int, Channel<MarketDataResponse>> _marketDataChannels = new();
    private readonly Dictionary<int, string> _tickerSymbols = new();

    public ReflectionIbkrProvider()
    {
        try
        {
            try
            {
                _signal = (EReaderSignal)Activator.CreateInstance(typeof(EReaderSignal));
            }
            catch
            {
                // If EReaderSignal is abstract or cannot be created, set to null
                _signal = null;
            }
            _wrapper = new IbkrWrapper(this);
            // Use reflection to create EClientSocket
            var ctor = typeof(EClientSocket).GetConstructor(new Type[] { typeof(EWrapper) });
            if (ctor != null)
            {
                _clientSocket = (EClientSocket)ctor.Invoke(new object[] { _wrapper });
            }
            else
            {
                var ctor2 = typeof(EClientSocket).GetConstructor(new Type[] { typeof(EWrapper), typeof(EReaderSignal) });
                if (ctor2 != null && _signal != null)
                {
                    _clientSocket = (EClientSocket)ctor2.Invoke(new object[] { _wrapper, _signal });
                }
                else
                {
                    throw new Exception("Cannot create EClientSocket with available constructors");
                }
            }
            if (_signal != null)
            {
                _reader = new EReader(_clientSocket, _signal);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error initializing IBKR provider: {ex.Message}");
            _clientSocket = null!;
            _signal = null;
            _reader = null;
            _wrapper = null!;
        }
    }

    public async Task<ConnectResponse> Connect(ConnectRequest request)
    {
        try
        {
            Console.WriteLine($"Connecting to IBKR at {request.Host}:{request.Port} with client ID {request.ClientId}");
            _clientSocket.eConnect(request.Host, request.Port, request.ClientId);
            await Task.Delay(2000); // Wait for connection to establish
            if (_clientSocket.IsConnected())
            {
                // Start the reader if available
                _reader?.Start();
                // Request next valid order ID
                _clientSocket.reqIds(-1);
                return new ConnectResponse { Success = true, Message = "Connected successfully" };
            }
            else
            {
                return new ConnectResponse { Success = false, Message = "Failed to connect" };
            }
        }
        catch (Exception ex)
        {
            return new ConnectResponse { Success = false, Message = $"Connection error: {ex.Message}" };
        }
    }

    public async Task<ConnectionStatusResponse> GetConnectionStatus(ConnectionStatusRequest request)
    {
        try
        {
            bool connected = _clientSocket.IsConnected();
            return new ConnectionStatusResponse { Connected = connected, Message = connected ? "Connected" : "Not connected" };
        }
        catch (Exception ex)
        {
            return new ConnectionStatusResponse { Connected = false, Message = $"Status check error: {ex.Message}" };
        }
    }

    public async Task StreamMarketData(MarketDataRequest request, IServerStreamWriter<MarketDataResponse> responseStream, CancellationToken cancellationToken)
    {
        if (!_clientSocket.IsConnected()) return;

        var contract = new Contract
        {
            Symbol = request.Symbol,
            SecType = "STK",
            Exchange = request.Exchange ?? "SMART",
            Currency = "USD"
        };

        var tickerId = new Random().Next(1000, 9999);
        _tickerSymbols[tickerId] = request.Symbol;
        var channel = Channel.CreateUnbounded<MarketDataResponse>();
        _marketDataChannels[tickerId] = channel;

        _clientSocket.reqMktData(tickerId, contract, "", false, false, null);

        await foreach (var data in channel.Reader.ReadAllAsync(cancellationToken))
        {
            await responseStream.WriteAsync(data);
        }

        _clientSocket.cancelMktData(tickerId);
        _marketDataChannels.Remove(tickerId);
        _tickerSymbols.Remove(tickerId);
    }

    public Task<AccountResponse> GetAccountInfo(AccountRequest request)
    {
        Console.WriteLine($"GetAccountInfo called for {request.AccountId}");
        return Task.FromResult(new AccountResponse
        {
            Account = "TestAccount",
            TotalCash = 1000.0,
            NetLiquidation = 10000.0,
            BuyingPower = 2000.0,
            AvailableFunds = 1500.0,
            EquityWithLoan = 10000.0,
            ExcessLiquidity = 500.0,
            DayTradesRemaining = 3,
            AccountType = "Test"
        });
    }

    public Task<PlaceOrderResponse> PlaceOrder(PlaceOrderRequest request)
    {
        if (!_clientSocket.IsConnected()) return Task.FromResult(new PlaceOrderResponse { Success = false, Message = "Not connected" });

        var contract = new Contract
        {
            Symbol = request.Symbol,
            SecType = "STK",
            Exchange = "SMART",
            Currency = "USD"
        };

        var order = new Order
        {
            Action = request.Action,
            TotalQuantity = request.Quantity,
            OrderType = "LMT",
            LmtPrice = request.Price
        };

        _clientSocket.placeOrder(_nextOrderId++, contract, order);

        return Task.FromResult(new PlaceOrderResponse
        {
            OrderId = (_nextOrderId - 1).ToString(),
            Success = true,
            Message = "Order placed"
        });
    }

    private class IbkrWrapper : DefaultEWrapper
    {
        private readonly ReflectionIbkrProvider _provider;

        public IbkrWrapper(ReflectionIbkrProvider provider)
        {
            _provider = provider;
        }

        public override void nextValidId(int orderId)
        {
            _provider._nextOrderId = orderId;
            Console.WriteLine($"Next valid order ID: {orderId}");
        }

        public override void tickPrice(int tickerId, int field, double price, TickAttrib attribs)
        {
            _provider.OnTickPrice(tickerId, field, price, attribs);
        }

        public override void tickSize(int tickerId, int field, decimal size)
        {
            _provider.OnTickSize(tickerId, field, size);
        }

        public override void updateAccountValue(string key, string value, string currency, string accountName)
        {
            _provider.OnUpdateAccountValue(key, value, currency, accountName);
        }

        public override void connectionClosed()
        {
            Console.WriteLine("Connection closed");
        }
    }

    private void OnTickPrice(int tickerId, int field, double price, TickAttrib attribs)
    {
        if (_marketDataChannels.TryGetValue(tickerId, out var channel))
        {
            var response = new MarketDataResponse
            {
                Symbol = _tickerSymbols.GetValueOrDefault(tickerId, ""),
                Price = price,
                Volume = 0 // Volume handled separately if needed
            };
            channel.Writer.TryWrite(response);
        }
    }

    private void OnTickSize(int tickerId, int field, decimal size)
    {
        if (field == 8 && _marketDataChannels.TryGetValue(tickerId, out var channel)) // Volume
        {
            // Update volume if needed, but for simplicity, skip
        }
    }

    private void OnUpdateAccountValue(string key, string value, string currency, string accountName)
    {
        if (key == "AccountType")
        {
            _accountType = value;
        }
        else if (double.TryParse(value, out double val))
        {
            _accountData[key] = val;
        }
    }
}

