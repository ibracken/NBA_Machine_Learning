import requests
import json
import pandas as pd

def test_nba_api():
    url = "https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&ISTRound=&LeagueID=00&PlayerOrTeam=P&Season=2024-25&SeasonType=Regular%20Season&Sorter=DATE"
    
    # Use your existing proxy
    proxy_url = "http://smart-b0ibmkjy90uq_area-US_state-Northcarolina_life-15_session-0Ve35bhsUr:sU8CQmV8LDmh2mXj@proxy.smartproxy.net:3120"
    proxies = {
        'http': proxy_url,
        'https': proxy_url
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nba.com/stats/players/advanced',
        'Origin': 'https://www.nba.com',
        'Host': 'stats.nba.com',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    try:
        print("Testing NBA API endpoint...")
        
        # Use session with proxy
        session = requests.Session()
        session.headers.update(headers)
        session.proxies.update(proxies)
        
        print("Making direct API request with proxy...")
        response = session.get(url, timeout=30, verify=False)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response Keys: {list(data.keys())}")
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                result_set = data['resultSets'][0]
                headers_list = result_set['headers']
                rows = result_set['rowSet']
                
                print(f"Headers: {headers_list}")
                print(f"Number of players: {len(rows)}")
                print(f"Sample row: {rows[0] if rows else 'No data'}")
                
                # Create DataFrame
                df = pd.DataFrame(rows, columns=headers_list)
                print(f"DataFrame shape: {df.shape}")
                print(f"All columns returned:")
                print(df.columns.tolist())
                
                return True
            else:
                print("No resultSets found in response")
                print(f"Full response: {json.dumps(data, indent=2)}")
                return False
        else:
            print(f"Request failed with status {response.status_code}")
            print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_nba_api()