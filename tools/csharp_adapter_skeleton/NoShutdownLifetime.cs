using Microsoft.Extensions.Hosting;

public class NoShutdownLifetime : IHostLifetime

{

    public Task StopAsync(CancellationToken cancellationToken) => Task.Delay(-1, cancellationToken);

    public Task WaitForStartAsync(CancellationToken cancellationToken)

    {

        Console.WriteLine("WaitForStartAsync called");

        return Task.CompletedTask;

    }

}