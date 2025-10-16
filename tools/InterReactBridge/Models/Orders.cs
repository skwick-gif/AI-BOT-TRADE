namespace InterReactBridge.Models;

public class BracketOrderRequest
{
    public string Symbol { get; set; } = "";
    public string SecType { get; set; } = "STK";
    public string Exchange { get; set; } = "SMART";
    public string Action { get; set; } = "BUY"; // BUY/SELL
    public int Quantity { get; set; }
    public string EntryType { get; set; } = "LMT"; // MKT/LMT
    public double? EntryPrice { get; set; }
    public double? TakeProfitPrice { get; set; }
    public double? StopLossPrice { get; set; }
    public bool OutsideRth { get; set; } = false;
}

public class OcoOrderItem
{
    public string Action { get; set; } = "SELL"; // BUY/SELL
    public string OrderType { get; set; } = "LMT"; // LMT/STP/MKT
    public int Quantity { get; set; }
    public double? Price { get; set; }
    public double? StopPrice { get; set; }
}

public class OcoOrderRequest
{
    public string Symbol { get; set; } = "";
    public string SecType { get; set; } = "STK";
    public string Exchange { get; set; } = "SMART";
    public string OcaGroup { get; set; } = ""; // optional; auto if empty
    public List<OcoOrderItem> Orders { get; set; } = new();
}

public class ComboLegItem
{
    public int ConId { get; set; }
    public int Ratio { get; set; } = 1;
    public string Action { get; set; } = "BUY"; // BUY/SELL
    public string Exchange { get; set; } = "SMART";
}

public class ComboOrderRequest
{
    public string Exchange { get; set; } = "SMART";
    public string Currency { get; set; } = "USD";
    public List<ComboLegItem> Legs { get; set; } = new();
    public string OrderType { get; set; } = "LMT"; // MKT/LMT
    public double? Price { get; set; }
    public int Quantity { get; set; } = 1;
}
