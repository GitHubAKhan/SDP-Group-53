import json
from alpaca_trader import AlpacaTrader

def lambda_handler(event, context):
    """Lambda function to get Alpaca account/positions information."""
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    }
    
    try:
        # DEBUG - Print the entire event
        print("Full event:", json.dumps(event))
        
        # Parse request body - handle multiple formats
        body = None
        
        # Format 1: event has 'body' as a string
        if 'body' in event and isinstance(event['body'], str):
            print("Parsing body as JSON string")
            body = json.loads(event['body'])
        # Format 2: event has 'body' as a dict
        elif 'body' in event and isinstance(event['body'], dict):
            print("Body is already a dict")
            body = event['body']
        # Format 3: credentials are directly in event
        elif 'apiKey' in event:
            print("Credentials directly in event")
            body = event
        else:
            print("Unknown format, using event as-is")
            body = event
        
        print("Parsed body:", json.dumps(body))
        
        # Get credentials
        api_key = body.get('apiKey')
        api_secret = body.get('apiSecret')
        
        print(f"API Key found: {api_key is not None}")
        print(f"API Secret found: {api_secret is not None}")
        
        if not api_key or not api_secret:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Missing apiKey or apiSecret',
                    'received_body': body,
                    'event_keys': list(event.keys())
                })
            }
        
        # Initialize trader and get data
        trader = AlpacaTrader(api_key=api_key, api_secret=api_secret)
        
        # FOR get_account:
        data = trader.get_account()
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(data)
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })
        }