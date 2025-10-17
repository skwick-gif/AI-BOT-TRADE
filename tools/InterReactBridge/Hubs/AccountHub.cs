using Microsoft.AspNetCore.SignalR;

namespace InterReactBridge.Hubs;

/// <summary>
/// SignalR Hub for real-time account updates
/// </summary>
public class AccountHub : Hub
{
    private readonly ILogger<AccountHub> _logger;

    public AccountHub(ILogger<AccountHub> logger)
    {
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        _logger.LogInformation("Client connected to AccountHub: {ConnectionId}", Context.ConnectionId);
        await base.OnConnectedAsync();
        
        // Send initial welcome message
        await Clients.Caller.SendAsync("Connected", new { connectionId = Context.ConnectionId });
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        _logger.LogInformation("Client disconnected from AccountHub: {ConnectionId}", Context.ConnectionId);
        await base.OnDisconnectedAsync(exception);
    }

    /// <summary>
    /// Client can request immediate account summary update
    /// </summary>
    public async Task RequestAccountUpdate()
    {
        _logger.LogInformation("Account update requested by: {ConnectionId}", Context.ConnectionId);
        // Background service will handle the actual update
        await Clients.Caller.SendAsync("AccountUpdateRequested", new { status = "processing" });
    }
}
