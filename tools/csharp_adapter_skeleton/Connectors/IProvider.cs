using Grpc.Core;
using IBKRAdapterServer.Protos;

namespace ExternalConnector;

public interface IProvider
{
    Task<ConnectResponse> Connect(ConnectRequest request);
    Task<ConnectionStatusResponse> GetConnectionStatus(ConnectionStatusRequest request);
    Task StreamMarketData(MarketDataRequest request, IServerStreamWriter<MarketDataResponse> responseStream, CancellationToken cancellationToken);
    Task<AccountResponse> GetAccountInfo(AccountRequest request);
    Task<PlaceOrderResponse> PlaceOrder(PlaceOrderRequest request);
}