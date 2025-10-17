using System;
using System.Threading.Tasks;
using InterReact;
using System.Reactive.Linq;

class TestConnect
{
    static async Task Main()
    {
        Console.WriteLine("=== Testing InterReact Connection ===");
        Console.WriteLine($"Time: {DateTime.Now}");
        Console.WriteLine("Host: 127.0.0.1");
        Console.WriteLine("Port: 7497");
        Console.WriteLine("ClientId: 0");
        Console.WriteLine();

        try
        {
            Console.WriteLine("Creating InterReact client...");
            var client = await InterReactClient.ConnectAsync(options =>
            {
                options.TwsIpAddress = System.Net.IPAddress.Parse("127.0.0.1");
                options.IBPortAddresses = new[] { 7497 };
                options.TwsClientId = 0;
            });

            Console.WriteLine("✓ Client created successfully!");
            Console.WriteLine("Waiting for ManagedAccounts...");

            var timeout = Task.Delay(10000);
            var accountTask = client.Response.OfType<ManagedAccounts>().FirstAsync().ToTask();

            var completed = await Task.WhenAny(accountTask, timeout);

            if (completed == accountTask)
            {
                var account = await accountTask;
                Console.WriteLine($"✓ Account received: {account.Accounts}");
            }
            else
            {
                Console.WriteLine("⚠ Timeout waiting for account info");
            }

            Console.WriteLine("\n=== Connection Test Complete ===");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n✗ Connection failed!");
            Console.WriteLine($"Error Type: {ex.GetType().Name}");
            Console.WriteLine($"Error Message: {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
            }
            Console.WriteLine($"\nStack Trace:\n{ex.StackTrace}");
        }

        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
}
