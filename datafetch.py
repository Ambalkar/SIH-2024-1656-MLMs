import requests

# Define the API URL and the authorization key
api_url = "https://gemini.incois.gov.in/OceanDataAPI/api/wqns/Kochi/chlorophyll"
authorization_key = "446d183e64e64e8eb4bca1407ab02a89"

# Set the headers with the authorization key
headers = {
    "Authorization": authorization_key
}

# Send a GET request to the API
response = requests.get(api_url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    print("Chlorophyll Data:", data)
else:
    print("Failed to retrieve data. Status code:", response.status_code)
