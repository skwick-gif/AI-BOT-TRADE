using IBKRAdapterServer.Services;

var builder = WebApplication.CreateBuilder(args);
    builder.WebHost.ConfigureKestrel(options =>
    {
        options.ListenLocalhost(50051, o => o.Protocols = Microsoft.AspNetCore.Server.Kestrel.Core.HttpProtocols.Http1AndHttp2);
    });builder.Services.AddGrpc();

var app = builder.Build();

app.MapGrpcService<IBKRServiceImpl>();

app.MapGet("/", () => "Hello");

app.Run();