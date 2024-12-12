from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def generate_token():
    creds = None
    
    # Create credentials directory if it doesn't exist
    os.makedirs('google_credentials', exist_ok=True)
    
    # The file token.pickle stores the user's access and refresh tokens
    token_path = 'google_credentials/token.pickle'
    credentials_path = 'google_credentials/credentials.json'
    
    # Check if credentials.json exists
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            "credentials.json not found. Please download it from Google Cloud Console "
            "and save it in the google_credentials directory."
        )
    
    if os.path.exists(token_path):
        print(f"Token file already exists at {token_path}")
        return
    
    # Create token
    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_path, SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Save the credentials for the next run
    with open(token_path, 'wb') as token:
        pickle.dump(creds, token)
    
    print(f"Successfully created token file at {token_path}")

if __name__ == '__main__':
    generate_token() 