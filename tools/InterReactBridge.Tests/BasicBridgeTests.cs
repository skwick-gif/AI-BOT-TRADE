using Microsoft.AspNetCore.Mvc.Testing;
using System.Net;
using System.Text.Json;
using Xunit;

namespace InterReactBridge.Tests
{
    public class BasicBridgeTests : IClassFixture<WebApplicationFactory<Program>>
    {
        private readonly WebApplicationFactory<Program> _factory;
        private readonly HttpClient _client;

        public BasicBridgeTests(WebApplicationFactory<Program> factory)
        {
            _factory = factory;
            _client = _factory.CreateClient();
        }

        [Fact]
        public async Task HealthEndpoint_ReturnsOk()
        {
            // Act
            var response = await _client.GetAsync("/health");

            // Assert
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            
            var content = await response.Content.ReadAsStringAsync();
            var healthResponse = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
            
            Assert.NotNull(healthResponse);
            Assert.True(healthResponse.ContainsKey("status"));
            Assert.Equal("ok", healthResponse["status"]?.ToString());
        }

        [Fact]
        public async Task ConnectEndpoint_WithMissingParameters_ReturnsBadRequest()
        {
            // Act - missing required parameters
            var response = await _client.PostAsync("/connect", null);

            // Assert
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        }

        [Fact]
        public async Task MarketDataEndpoint_WithoutSymbol_ReturnsBadRequest()
        {
            // Act
            var response = await _client.GetAsync("/marketdata");

            // Assert
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        }

        [Fact]
        public async Task AccountEndpoint_WithoutConnection_ReturnsError()
        {
            // Act
            var response = await _client.GetAsync("/account");

            // Assert
            var content = await response.Content.ReadAsStringAsync();
            Assert.NotNull(content);
            
            // Should return error status since we're not connected
            Assert.True(response.StatusCode == HttpStatusCode.BadRequest || 
                       response.StatusCode == HttpStatusCode.InternalServerError);
        }

        [Fact]
        public async Task PortfolioEndpoint_WithoutConnection_ReturnsError()
        {
            // Act
            var response = await _client.GetAsync("/portfolio");

            // Assert
            var content = await response.Content.ReadAsStringAsync();
            Assert.NotNull(content);
            
            // Should return error status since we're not connected
            Assert.True(response.StatusCode == HttpStatusCode.BadRequest || 
                       response.StatusCode == HttpStatusCode.InternalServerError);
        }
    }
}