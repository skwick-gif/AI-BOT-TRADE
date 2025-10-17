namespace InterReactBridge.Services.Indicators;

/// <summary>
/// Base interface for technical indicators
/// </summary>
public interface IIndicator
{
    /// <summary>
    /// Name of the indicator (e.g., "SMA", "RSI", "MACD")
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Add a new price point to the indicator
    /// </summary>
    /// <param name="price">Price value</param>
    /// <param name="timestamp">Time of the price</param>
    void AddPrice(decimal price, DateTime timestamp);

    /// <summary>
    /// Calculate and return the current indicator value
    /// Returns null if not enough data
    /// </summary>
    decimal? Calculate();

    /// <summary>
    /// Reset the indicator (clear all data)
    /// </summary>
    void Reset();

    /// <summary>
    /// Check if indicator has enough data to calculate a value
    /// </summary>
    bool IsReady { get; }
}
