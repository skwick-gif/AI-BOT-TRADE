// See https://aka.ms/new-console-template for more information
Console.WriteLine("Testing IBKR DLL...");

try
{
    var client = new IBApi.EClientSocket(null, null);
    Console.WriteLine("EClientSocket created successfully.");
}
catch (Exception ex)
{
    Console.WriteLine($"Error creating EClientSocket: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
}
