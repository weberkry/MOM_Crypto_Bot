# MCP Agent stub

class MCPAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def run_pipeline(self, symbol: str):
        """Return a dict with sentiment_score [-1,1], summary string, and topics list"""
        # TODO: replace with actual MCP client logic
        # Example return format:
        return {
            'sentiment_score': 0.0,
            'summary': f'No MCP configured. Placeholder summary for {symbol}',
            'topics': []
        }
