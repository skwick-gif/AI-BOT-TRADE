using InterReactBridge.Services;

var builder = WebApplication.CreateBuilder(args);

// הזרקת תלות
builder.Services.AddLogging();
builder.Services.AddSingleton<IbService>();

// קבע את ה-port ל-5080
builder.WebHost.UseUrls("http://localhost:5080");

var app = builder.Build();

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
// Start the server
// -----------------------------
app.Run();
