using InterReactBridge.Services;

var builder = WebApplication.CreateBuilder(args);

// הזרקת תלות
// builder.Services.AddSingleton<IbService>();  // Temporarily disabled
builder.Services.AddLogging();

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
app.MapGet("/marketdata", async (/*IbService ib,*/ string symbol, string secType, string exchange, int durationSeconds) =>
{
    return Results.Ok(new { message = "IbService disabled for testing" });
});

// -----------------------------
// Connect to IBKR
// דוגמה: POST /connect?host=127.0.0.1&port=7496&clientId=1
// -----------------------------
app.MapPost("/connect", async (/*IbService ib,*/ string host, int port, int clientId) =>
{
    return Results.Ok(new { connected = false, message = "IbService disabled for testing" });
});

// -----------------------------
// Account summary
// דוגמה: GET /account
// -----------------------------
app.MapGet("/account", async (/*IbService ib*/) =>
{
    return Results.Ok(new { message = "IbService disabled for testing" });
});

// -----------------------------
// Portfolio positions
// דוגמה: GET /portfolio
// -----------------------------
app.MapGet("/portfolio", async (/*IbService ib*/) =>
{
    return Results.Ok(new { message = "IbService disabled for testing" });
});

// -----------------------------
// Place order
// דוגמה: POST /order?symbol=AAPL&secType=STK&exchange=SMART&action=BUY&quantity=100&price=150&orderType=LMT
// -----------------------------
app.MapPost("/order", async (/*IbService ib,*/ string symbol, string secType, string exchange, string action, int quantity, double? price, string? orderType) =>
{
    return Results.Ok(new { message = "IbService disabled for testing" });
});

// -----------------------------
// Options chain
// דוגמה: GET /optionschain?underlying=AAPL&exchange=SMART
// -----------------------------
app.MapGet("/optionschain", async (/*IbService ib,*/ string underlying, string? exchange) =>
{
    return Results.Ok(new { message = "IbService disabled for testing" });
});

// -----------------------------
// Live market data (Server-Sent Events)
// דוגמה: GET /livedata?symbol=AAPL&secType=STK&exchange=SMART
// -----------------------------
app.MapGet("/livedata", async (/*IbService ib,*/ HttpResponse response, string symbol, string secType, string exchange) =>
{
    await response.WriteAsync("data: {\"message\": \"IbService disabled for testing\"}\n\n");
});

// -----------------------------
// Scanner
// דוגמה: GET /scan?scanType=TOP_PERC_GAIN&numberOfRows=10
// -----------------------------
app.MapGet("/scan", async (/*IbService ib,*/ string? scanType, int? numberOfRows) =>
{
    return Results.Ok(new { message = "IbService disabled for testing" });
});

// -----------------------------
// Start the server
// -----------------------------
app.Run();
