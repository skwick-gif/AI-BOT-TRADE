namespace InterReactBridge.Services.Indicators;

/// <summary>
/// Simple Moving Average (SMA) Indicator
/// Calculates the average of the last N prices
/// 
/// Formula: SMA = (P1 + P2 + ... + Pn) / n
/// where P = price, n = period
/// 
/// Example: SMA(20) = average of last 20 prices
/// </summary>
public class SimpleMovingAverage : IIndicator
{
    private readonly int _period;
    private readonly Queue<decimal> _prices;

    public string Name => $"SMA({_period})";

    /// <summary>
    /// Create a Simple Moving Average indicator
    /// </summary>
    /// <param name="period">Number of periods to average (e.g., 20 for SMA-20)</param>
    public SimpleMovingAverage(int period)
    {
        if (period <= 0)
            throw new ArgumentException("Period must be greater than 0", nameof(period));

        _period = period;
        _prices = new Queue<decimal>(period);
    }

    public void AddPrice(decimal price, DateTime timestamp)
    {
        _prices.Enqueue(price);

        // Keep only the last N prices
        while (_prices.Count > _period)
        {
            _prices.Dequeue();
        }
    }

    public decimal? Calculate()
    {
        if (!IsReady)
            return null;

        // Calculate average
        return _prices.Average();
    }

    public void Reset()
    {
        _prices.Clear();
    }

    public bool IsReady => _prices.Count >= _period;

    /// <summary>
    /// Get current number of data points
    /// </summary>
    public int Count => _prices.Count;

    /// <summary>
    /// Get the required period for this indicator
    /// </summary>
    public int Period => _period;
}
