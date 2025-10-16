using InterReactBridge.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Builder;

var builder = WebApplication.CreateBuilder(args);

// Configure Kestrel/URLs
// Preference order:
// 1) ASPNETCORE_URLS env (standard)
// 2) IBKR_BRIDGE_URL env (custom)
// 3) --urls passed at runtime (handled by framework)
// 4) fallback to http://localhost:8080
var envUrl = Environment.GetEnvironmentVariable("ASPNETCORE_URLS")
    ?? Environment.GetEnvironmentVariable("IBKR_BRIDGE_URL");

if (!string.IsNullOrWhiteSpace(envUrl))
{
    builder.WebHost.UseUrls(envUrl);
}
else
{
    // If no env provided and no --urls, set default
    // Note: --urls is automatically handled by the host builder, so we don't need to parse args here.
    builder.WebHost.UseUrls("http://localhost:8080");
}

// Dependency injection
builder.Services.AddSingleton<IbService>();
builder.Services.AddLogging();

var app = builder.Build();

// -----------------------------
// Health check
// -----------------------------
app.MapGet("/health", () => Results.Ok(new { status = "ok", time = DateTime.UtcNow }));

// Diagnostics: full status
app.MapGet("/diagnostics", (IbService ib) => Results.Ok(ib.GetStatus()));
app.MapGet("/connect/status", (IbService ib) => Results.Ok(ib.GetStatus()));
app.MapGet("/probe", async (IbService ib, string host, int port, int? timeoutMs) =>
{
    var res = await ib.ProbeTcpAsync(host, port, timeoutMs ?? 800);
    return Results.Ok(res);
});

// -----------------------------
// Connect to IBKR
// Example: POST /connect?host=127.0.0.1&port=7496&clientId=1
// -----------------------------
app.MapPost("/connect", async (IbService ib, string host, int port, int clientId) =>
{
    try
    {
        var connected = await ib.ConnectAsync(host, port, clientId);
        return Results.Ok(new { connected, host, port, clientId });
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Connect failed", detail: ex.Message, statusCode: 500);
    }
});

// Retry last connect attempt with previous host/port/clientId
app.MapPost("/connect/retry", async (IbService ib) =>
{
    try
    {
        var (host, port, clientId) = ib.GetLastEndpoint();
        if (string.IsNullOrWhiteSpace(host) || port <= 0)
        {
            return Results.BadRequest(new { success = false, message = "No previous connect attempt found" });
        }
        var connected = await ib.ConnectAsync(host!, port, clientId);
        return Results.Ok(new { connected, host, port, clientId });
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Retry connect failed", detail: ex.Message, statusCode: 500);
    }
});

// -----------------------------
// Account summary
// Example: GET /account
// -----------------------------
app.MapGet("/account", async (IbService ib) =>
{
    try
    {
        var data = await ib.GetAccountSummary();
        return Results.Ok(data);
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Account fetch failed", detail: ex.Message, statusCode: 500);
    }
});

// -----------------------------
// Portfolio positions
// Example: GET /portfolio
// -----------------------------
app.MapGet("/portfolio", async (IbService ib) =>
{
    try
    {
        var data = await ib.GetPortfolio();
        return Results.Ok(data);
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Portfolio fetch failed", detail: ex.Message, statusCode: 500);
    }
});

// -----------------------------
// Market data (one-shot)
// Example: GET /marketdata?symbol=EUR&secType=CASH&exchange=IDEALPRO&durationSeconds=5
// -----------------------------
app.MapGet("/marketdata", async (IbService ib, string symbol, string secType, string exchange, int durationSeconds) =>
{
    try
    {
        var data = await ib.GetMarketData(symbol, secType, exchange, TimeSpan.FromSeconds(Math.Max(1, durationSeconds)));
        return Results.Ok(data);
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Market data failed", detail: ex.Message, statusCode: 500);
    }
});

// -----------------------------
// Place order
// Example: POST /order?symbol=AAPL&secType=STK&exchange=SMART&action=BUY&quantity=100&price=150&orderType=LMT
// -----------------------------
app.MapPost("/order", async (IbService ib, string symbol, string secType, string exchange, string action, int quantity, double? price, string? orderType) =>
{
    try
    {
        var result = await ib.PlaceOrderAsync(symbol, secType, exchange, action, quantity, price, orderType ?? "LMT");
        return Results.Ok(new { success = true, result });
    }
    catch (Exception ex)
    {
        return Results.Ok(new { success = false, message = ex.Message });
    }
});

// -----------------------------
// Options chain
// Example: GET /optionschain?underlying=AAPL&exchange=SMART
// -----------------------------
app.MapGet("/optionschain", async (IbService ib, string underlying, string? exchange) =>
{
    try
    {
        var data = await ib.GetOptionsChainAsync(underlying, exchange ?? "SMART");
        return Results.Ok(data);
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Options chain failed", detail: ex.Message, statusCode: 500);
    }
});

// -----------------------------
// Live market data (Server-Sent Events)
// Example: GET /livedata?symbol=AAPL&secType=STK&exchange=SMART
// -----------------------------
app.MapGet("/livedata", async (IbService ib, HttpResponse response, string symbol, string secType, string exchange) =>
{
    try
    {
        await ib.GetLiveMarketData(response, symbol, secType, exchange);
    }
    catch (Exception ex)
    {
        await response.WriteAsync($"event: error\n" + $"data: {System.Text.Json.JsonSerializer.Serialize(new { message = ex.Message })}\n\n");
    }
});

// -----------------------------
// Scanner
// Example: GET /scan?scanType=TOP_PERC_GAIN&numberOfRows=10
// -----------------------------
app.MapGet("/scan", async (IbService ib, string? scanType, int? numberOfRows) =>
{
    try
    {
        var data = await ib.GetScannerAsync(scanType ?? "TOP_PERC_GAIN", numberOfRows ?? 10);
        return Results.Ok(data);
    }
    catch (Exception ex)
    {
        return Results.Problem(title: "Scanner failed", detail: ex.Message, statusCode: 500);
    }
});

// -----------------------------
// Start the server
// -----------------------------
// Optional AutoConnect on startup (IBKR_AUTO_CONNECT=true)
var auto = Environment.GetEnvironmentVariable("IBKR_AUTO_CONNECT");
if (!string.IsNullOrWhiteSpace(auto) && (auto.Equals("true", StringComparison.OrdinalIgnoreCase) || auto == "1"))
{
    var host = Environment.GetEnvironmentVariable("IBKR_HOST") ?? "127.0.0.1";
    var portStr = Environment.GetEnvironmentVariable("IBKR_PORT") ?? "4002";
    var clientIdStr = Environment.GetEnvironmentVariable("IBKR_CLIENT_ID") ?? "1";
    _ = Task.Run(async () =>
    {
        try
        {
            if (int.TryParse(portStr, out var port) && int.TryParse(clientIdStr, out var clientId))
            {
                var ib = app.Services.GetRequiredService<IbService>();
                await ib.ConnectAsync(host, port, clientId);
            }
        }
        catch { }
    });
}
app.Run();
