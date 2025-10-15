import grpc
import ibkr_pb2
import ibkr_pb2_grpc

class IBKRAdapterClient:
    def __init__(self, host='localhost', port=7000):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = ibkr_pb2_grpc.IBKRServiceStub(self.channel)

    def connect(self, host='127.0.0.1', port=7497, client_id=1):
        request = ibkr_pb2.ConnectRequest(host=host, port=port, client_id=client_id)
        try:
            response = self.stub.Connect(request)
            return response
        except grpc.RpcError as e:
            print(f"Error connecting: {e}")
            return None

    def get_connection_status(self):
        request = ibkr_pb2.ConnectionStatusRequest()
        try:
            response = self.stub.GetConnectionStatus(request)
            return response
        except grpc.RpcError as e:
            print(f"Error getting connection status: {e}")
            return None

    def stream_market_data(self, symbol, exchange=''):
        request = ibkr_pb2.MarketDataRequest(symbol=symbol, exchange=exchange)
        try:
            for response in self.stub.StreamMarketData(request):
                yield response
        except grpc.RpcError as e:
            print(f"Error streaming market data: {e}")

    def get_account_info(self, account_id):
        request = ibkr_pb2.AccountRequest(account_id=account_id)
        try:
            response = self.stub.GetAccountInfo(request)
            return response
        except grpc.RpcError as e:
            print(f"Error getting account info: {e}")
            return None

    def place_order(self, symbol, action, quantity, price):
        request = ibkr_pb2.PlaceOrderRequest(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price
        )
        try:
            response = self.stub.PlaceOrder(request)
            return response
        except grpc.RpcError as e:
            print(f"Error placing order: {e}")
            return None

    def close(self):
        self.channel.close()