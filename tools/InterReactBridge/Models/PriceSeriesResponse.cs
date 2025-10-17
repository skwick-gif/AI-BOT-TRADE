namespace InterReactBridge.Models;

/// <summary>
/// Response model for delayed price series
/// </summary>
public class PriceSeriesResponse
{
    public string Symbol { get; set; } = string.Empty;
    public int SampleSeconds { get; set; }
    public int PricesCount { get; set; }
    public List<PricePoint> Prices { get; set; } = new();
    public string Note { get; set; } = string.Empty;
}

/// <summary>
/// Single price point in the series
/// </summary>
public class PricePoint
{
    public double Price { get; set; }
    public string TickType { get; set; } = string.Empty;
    public DateTime Time { get; set; }
}
