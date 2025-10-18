using InterReactBridge.Services.Indicators;

/// <summary>
/// Manual tests for technical indicators
/// Run this to verify indicator calculations are correct
/// </summary>
public class IndicatorTests
{
    public static void TestSimpleMovingAverage()
    {
        Console.WriteLine("=== Testing Simple Moving Average (SMA) ===\n");

        // Test 1: SMA with period 3
        Console.WriteLine("Test 1: SMA(3) with prices [10, 20, 30, 40, 50]");
        var sma3 = new SimpleMovingAverage(3);

        Console.WriteLine($"Name: {sma3.Name}");
        Console.WriteLine($"IsReady: {sma3.IsReady} (Expected: False)");
        Console.WriteLine($"Count: {sma3.Count}/3");
        Console.WriteLine();

        // Add prices one by one
        var prices = new[] { 10m, 20m, 30m, 40m, 50m };
        foreach (var price in prices)
        {
            sma3.AddPrice(price, DateTime.Now);
            var value = sma3.Calculate();
            Console.WriteLine($"Added price: {price,5}, IsReady: {sma3.IsReady}, Count: {sma3.Count}, SMA = {value?.ToString("F2") ?? "N/A"}");
        }

        Console.WriteLine();
        Console.WriteLine("Expected results:");
        Console.WriteLine("  After 10: N/A (not enough data)");
        Console.WriteLine("  After 20: N/A (not enough data)");
        Console.WriteLine("  After 30: 20.00 (average of 10, 20, 30)");
        Console.WriteLine("  After 40: 30.00 (average of 20, 30, 40)");
        Console.WriteLine("  After 50: 40.00 (average of 30, 40, 50)");
        Console.WriteLine();

        // Test 2: Reset functionality
        Console.WriteLine("Test 2: Reset functionality");
        Console.WriteLine($"Before reset - Count: {sma3.Count}, IsReady: {sma3.IsReady}");
        sma3.Reset();
        Console.WriteLine($"After reset  - Count: {sma3.Count}, IsReady: {sma3.IsReady}");
        Console.WriteLine();

        // Test 3: SMA with period 5
        Console.WriteLine("Test 3: SMA(5) with prices [100, 105, 110, 115, 120]");
        var sma5 = new SimpleMovingAverage(5);
        var prices2 = new[] { 100m, 105m, 110m, 115m, 120m };
        
        foreach (var price in prices2)
        {
            sma5.AddPrice(price, DateTime.Now);
        }

        var result = sma5.Calculate();
        Console.WriteLine($"SMA(5) = {result:F2}");
        Console.WriteLine($"Expected: 110.00 (average of 100, 105, 110, 115, 120)");
        Console.WriteLine();

        Console.WriteLine("=== All Tests Complete ===");
    }

    // Main method temporarily disabled to prevent conflicts with Program.cs
    /*
    public static void Main(string[] args)
    {
        try
        {
            TestSimpleMovingAverage();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }

        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
    */
}
