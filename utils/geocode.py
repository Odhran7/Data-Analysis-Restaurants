import requests
import time

# You need a Geoapify API key to call the Batch Geocoding API (https://www.geoapify.com/solutions/batch-geocoding-requests)
# Sign up on https://www.geoapify.com/ and generate an API key
# The Free Plan (https://www.geoapify.com/pricing/) lets you geocode up to 6000 addresses/day
# Replace "YOUR_API_KEY" with API key
apiKey = "YOUR_API_KEY"

# With Batch Geocoding, you create a geocoding job by sending addresses and then, after some time, get geocoding results by job id
# You may require a few attempts to get results. Here is a timeout between the attempts - 1 sec. Increase the timeout for larger jobs.
timeout = 1

# Limit the number of attempts
maxAttempt = 10

def getLocations(locations):
    url = "https://api.geoapify.com/v1/batch/geocode/search?apiKey=" + apiKey
    response = requests.post(url, json = locations)
    result = response.json()

    # The API returns the status code 202 to indicate that the job was accepted and pending
    status = response.status_code
    if (status != 202):
        print('Failed to create a job. Check if the input data is correct.')
        return
    jobId = result['id']
    getResultsUrl = url + '&id=' + jobId

    time.sleep(timeout)
    result = getLocationJobs(getResultsUrl, 0)
    if (result):
        print(result)
        print('You can also get results by the URL - ' + getResultsUrl)
    else:
        print('You exceeded the maximal number of attempts. Try to get results later. You can do this in a browser by the URL - ' + getResultsUrl)

def getLocationJobs(url, attemptCount):
    response = requests.get(url)
    result = response.json()
    status = response.status_code
    if (status == 200):
        print('The job is succeeded. Here are the results:')
        return result
    elif (attemptCount >= maxAttempt):
        return
    elif (status == 202):
        print('The job is pending...')
        time.sleep(timeout)
        return getLocationJobs(url, attemptCount + 1)

# Addresses to geocode
data = [
    "668 Cedar St, San Carlos, CA 94070, United States of America",
    "545 Southwest Taylor Street, Portland, OR 97204, United States of America",
    "1415 Southwest Park Avenue, Portland, OR 97201, United States of America",
    "1019 Southwest Morrison Street, Portland, OR 97205, United States of America",
    "400 Southwest 6th Avenue, Portland, OR 97204, United States of America",
    "1972 Northwest Flanders Street, Portland, OR 97209, United States of America",
    "1150 Northwest Quimby Street, Portland, OR 97209, United States of America",
    "2116 Northwest 20th Avenue, Portland, OR 97209, United States of America",
    "200 East 13th Street, Vancouver, WA 98660, United States of America",
    "1108 Main St, Vancouver, WA 98660, United States of America",
    "410 West Mill Plain Boulevard, Vancouver, WA 98660, United States of America",
    "900 W Evergreen Blvd, Vancouver, WA 98660, United States of America",
    "2008 Simpson Ave, Vancouver, WA 98660, United States of America"
]
getLocations(data)