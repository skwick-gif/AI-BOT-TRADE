using Microsoft.AspNetCore.SignalR;

namespace InterReactBridge.Hubs;

/// <summary>
/// SignalR Hub for real-time portfolio/positions updates
/// </summary>
public class PortfolioHub : Hub
{
    private readonly ILogger<PortfolioHub> _logger;

    public PortfolioHub(ILogger<PortfolioHub> logger)
    {
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        _logger.LogInformation("Client connected to PortfolioHub: {ConnectionId}", Context.ConnectionId);
        await base.OnConnectedAsync();
        
        await Clients.Caller.SendAsync("Connected", new { connectionId = Context.ConnectionId });
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _logger.LogInformation("Client disconnected from PortfolioHub: {ConnectionId}", Context.ConnectionId);
        await base.OnDisconnectedAsync(exception);
    }

    /// <summary>
    /// Client can request immediate portfolio update
    /// </summary>
    public async Task RequestPortfolioUpdate()
    {
        _logger.LogInformation("Portfolio update requested by: {ConnectionId}", Context.ConnectionId);
        await Clients.Caller.SendAsync("PortfolioUpdateRequested", new { status = "processing" });
    }
}
