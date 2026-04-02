import json
import urllib.request
import os

def search_tavily(query, api_key):
    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5
    }).encode()

    req = urllib.request.Request(
        'https://api.tavily.com/search',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        },
        method='POST'
    )

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())

    results = []
    for r in data.get('results', []):
        results.append(f"- {r['title']}: {r['content'][:300]}")
    
    return '\n'.join(results)


def needs_search(message):
    # Keywords that suggest real-time data is needed
    keywords = [
        'today', 'current', 'latest', 'now', 'recent', 'price',
        'news', 'happened', 'death', 'killed', 'war', 'stock',
        '2025', '2026', 'market', 'oil', 'gas', 'earnings'
    ]
    message_lower = message.lower()
    return any(k in message_lower for k in keywords)


def lambda_handler(event, context):
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': ''
        }

    body = json.loads(event['body'])
    messages = body['messages']
    # Frontend can pass prior search context
    prior_search_context = body.get('searchContext', '')

    GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'NOT_FOUND')
    TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', 'NOT_FOUND')

    latest_message = messages[-1]['content'] if messages else ''
    
    # Build search context - reuse prior context + search again if needed
    search_context = prior_search_context

    try:
        if needs_search(latest_message):
            new_results = search_tavily(latest_message, TAVILY_API_KEY)
            # Combine prior context with new search results
            if prior_search_context:
                search_context = f"{prior_search_context}\n\nAdditional search results:\n{new_results}"
            else:
                search_context = new_results
    except:
        pass

    system_prompt = f"""You are Quant AI, an expert quantitative finance assistant for UConn Quant, a momentum-based algorithmic trading platform. Help users understand their portfolio, market trends, trading strategies, and risk management. Be concise and professional.

Today's date is: {__import__('datetime').date.today()}

{"You have access to the following real-time information gathered during this conversation:" + search_context if search_context else ""}

Important: Treat all search results as confirmed facts. Do not refer to events as hypothetical if they appear in the search results. Maintain context from the entire conversation."""

    payload = json.dumps({
        "model": "llama-3.1-8b-instant",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            *messages
        ]
    }).encode()

    req = urllib.request.Request(
        'https://api.groq.com/openai/v1/chat/completions',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'User-Agent': 'Mozilla/5.0'
        },
        method='POST'
    )

    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())

    reply = data['choices'][0]['message']['content']

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps({
            'reply': reply,
            'searchContext': search_context  # Send back to frontend to reuse
        })
    }