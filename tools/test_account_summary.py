import grpc
import ibkr_pb2
import ibkr_pb2_grpc

def test_account_summary():
    # Create a gRPC channel to the C# server
    channel = grpc.insecure_channel('localhost:50051')
    stub = ibkr_pb2_grpc.IBKRServiceStub(channel)

    # Create an empty request for GetAccountInfo
    request = ibkr_pb2.AccountRequest()

    try:
        # Call the GetAccountInfo method
        response = stub.GetAccountInfo(request)
        print("Account Summary Response:")
        print(f"Account: {response.account}")
        print(f"Total Cash: {response.totalCash}")
        print(f"Net Liquidation: {response.netLiquidation}")
        print(f"Buying Power: {response.buyingPower}")
        print(f"Available Funds: {response.availableFunds}")
        print(f"Equity With Loan: {response.equityWithLoan}")
        print(f"Excess Liquidity: {response.excessLiquidity}")
        print(f"Day Trades Remaining: {response.dayTradesRemaining}")
        print(f"Account Type: {response.accountType}")
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_account_summary()