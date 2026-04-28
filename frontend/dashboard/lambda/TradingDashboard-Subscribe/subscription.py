import json
import boto3
import re
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')

BUCKET_NAME = "trading-dashboard-sdteam-53"

def lambda_handler(event, context):

    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
    }

    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'message': 'CORS preflight successful'})
        }

    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})

        email = body.get('email', '').strip().lower()
        print(f"Subscribe attempt for: {email}")

        # Validate email
        if not validate_email(email):
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'success': False,
                    'message': 'Please enter a valid email address'
                })
            }

        # Read existing subscribers from S3
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key='subscribers.json')
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
            print(f"Loaded subscribers from S3: {data}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print("subscribers.json not found in S3, creating fresh list")
                data = {'subscribers': []}
            else:
                raise

        # Ensure subscribers key exists and all entries are plain strings
        if 'subscribers' not in data:
            data['subscribers'] = []

        # Normalize all existing entries to lowercase strings
        data['subscribers'] = [
            s.get('email', '').lower() if isinstance(s, dict) else str(s).lower()
            for s in data['subscribers']
        ]

        print(f"Current subscribers: {data['subscribers']}")

        # Check if already subscribed
        if email in data['subscribers']:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'success': True,
                    'message': f'{email} is already subscribed to notifications',
                    'alreadySubscribed': True
                })
            }

        # Add new subscriber
        data['subscribers'].append(email)
        print(f"Subscribers after append: {data['subscribers']}")

        # Write back to S3
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key='subscribers.json',
            Body=json.dumps(data, indent=2),
            ContentType='application/json'
        )

        print(f"Successfully wrote updated subscribers to S3")

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'success': True,
                'message': f'Successfully subscribed! You will receive notifications at {email}',
                'email': email
            })
        }

    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"AWS ClientError ({error_code}): {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'success': False,
                'message': 'Error saving subscription. Please try again.'
            })
        }

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'success': False,
                'message': 'An unexpected error occurred. Please try again.'
            })
        }


def validate_email(email):
    if not email or len(email) < 5 or len(email) > 320:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None