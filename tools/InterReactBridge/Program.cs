using InterReactBridge.Services;
using InterReactBridge.Hubs;

var builder = WebApplication.CreateBuilder(args);

// הזרקת תלות
builder.Services.AddLogging();
builder.Services.AddSingleton<IbService>();

// Background Service - The heart of production!
// Maintains persistent connection to TWS and broadcasts real-time updates
builder.Services.AddHostedService<TwsConnectionService>();

// SignalR for real-time streaming
builder.Services.AddSignalR();

// CORS policy for web clients
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// קבע את ה-port ל-5080
builder.WebHost.UseUrls("http://localhost:5080");

var app = builder.Build();

// -----------------------------
// Root
// -----------------------------
app.MapGet("/", () => "InterReactBridge is running");

// -----------------------------
// Health check
// -----------------------------
app.MapGet("/health", () =>
{
    return Results.Ok(new { status = "ok" });
});

// -----------------------------
// Market data (one-shot)
// Example: GET /marketdata?symbol=EUR&secType=CASH&exchange=IDEALPRO&durationSeconds=5
// -----------------------------
app.MapGet("/marketdata", async (IbService ib, string symbol, string secType, string exchange, int durationSeconds) =>
{
    try
    {
        var ticks = await ib.GetMarketData(symbol, secType, exchange, TimeSpan.FromSeconds(durationSeconds));
        return Results.Ok(ticks);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Connect to IBKR
// דוגמה: POST /connect?host=127.0.0.1&port=7496&clientId=1
// -----------------------------
app.MapPost("/connect", async (IbService ib, string host, int port, int clientId) =>
{
    var ok = await ib.ConnectAsync(host, port, clientId);
    return ok
        ? Results.Ok(new { connected = true })
        : Results.BadRequest(new { connected = false });
});

// -----------------------------
// Account summary
// דוגמה: GET /account
// -----------------------------
app.MapGet("/account", async (IbService ib) =>
{
    try
    {
        var account = await ib.GetAccountSummary();
        return Results.Ok(account);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// Portfolio positions
// דוגמה: GET /portfolio
// -----------------------------
app.MapGet("/portfolio", async (IbService ib) =>
{
    try
    {
        var portfolio = await ib.GetPortfolio();
        return Results.Ok(portfolio);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message });
    }
});

// -----------------------------
// SignalR Hubs for real-time streaming
// -----------------------------
app.UseCors("AllowAll");
app.MapHub<AccountHub>("/hubs/account");
app.MapHub<PortfolioHub>("/hubs/portfolio");
app.MapHub<MarketDataHub>("/hubs/marketdata");

// -----------------------------
// Start the server
// -----------------------------
Console.WriteLine("Starting InterReactBridge...");
Console.WriteLine("SignalR Hubs:");
Console.WriteLine("  - /hubs/account");
Console.WriteLine("  - /hubs/portfolio");
Console.WriteLine("  - /hubs/marketdata");
try
{
    app.Run();
}
catch (Exception ex)
{
    Console.WriteLine($"Application failed to start: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
}
