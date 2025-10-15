using Grpc.Core;
using IBKRAdapterServer.Protos;
using ExternalConnector;

namespace IBKRAdapterServer.Services;

public class IBKRServiceImpl : IBKRService.IBKRServiceBase
{
    private IProvider? _provider;

    public IBKRServiceImpl()
    {
        Console.WriteLine("IBKRServiceImpl constructor called");
        try
        {
            _provider = new ReflectionIbkrProvider();
            Console.WriteLine("Provider created successfully");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error creating provider: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw;
        }
    }

    public override async Task<ConnectResponse> Connect(ConnectRequest request, ServerCallContext context)
    {
        Console.WriteLine($"Connecting to IBKR at {request.Host}:{request.Port} with client ID {request.ClientId}");
        if (_provider != null)
        {
            return await _provider.Connect(request);
        }
        return new ConnectResponse { Success = false, Message = "Provider not available" };
    }

    public override async Task<ConnectionStatusResponse> GetConnectionStatus(ConnectionStatusRequest request, ServerCallContext context)
    {
        Console.WriteLine("Getting connection status");
        if (_provider != null)
        {
            return await _provider.GetConnectionStatus(request);
        }
        return new ConnectionStatusResponse { Connected = false, Message = "Provider not available" };
    }

    public override async Task StreamMarketData(MarketDataRequest request, IServerStreamWriter<MarketDataResponse> responseStream, ServerCallContext context)
    {
        Console.WriteLine($"Starting stream for {request.Symbol}");
        if (_provider != null)
        {
            await _provider.StreamMarketData(request, responseStream, context.CancellationToken);
        }
    }

    public override async Task<AccountResponse> GetAccountInfo(AccountRequest request, ServerCallContext context)
    {
        Console.WriteLine($"Getting account info for {request.AccountId}");
        try
        {
            if (_provider != null)
            {
                var result = await _provider.GetAccountInfo(request);
                Console.WriteLine($"Account info retrieved: {result.Account}");
                return result;
            }
            return new AccountResponse { Account = request.AccountId, TotalCash = 0, NetLiquidation = 0, BuyingPower = 0, AvailableFunds = 0, EquityWithLoan = 0, ExcessLiquidity = 0, DayTradesRemaining = 0, AccountType = "No Provider" };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in GetAccountInfo service: {ex.Message}");
            return new AccountResponse { Account = request.AccountId, TotalCash = 0, NetLiquidation = 0, BuyingPower = 0, AvailableFunds = 0, EquityWithLoan = 0, ExcessLiquidity = 0, DayTradesRemaining = 0, AccountType = $"Service Error: {ex.Message}" };
        }
    }

    public override async Task<PlaceOrderResponse> PlaceOrder(PlaceOrderRequest request, ServerCallContext context)
    {
        Console.WriteLine($"Placing order for {request.Symbol}");
        if (_provider != null)
        {
            return await _provider.PlaceOrder(request);
        }
        return new PlaceOrderResponse { Success = false, Message = "Provider not available" };
    }
}