# This file is used to scrape data from TripAdvisor

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.options import Options
import time
import json
import csv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='logs/tripadvisor_scraper.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Declare output file
output_csv = './output/extraMichelinNY.csv'

# Function parses the ratings by category
def get_ratings(driver):
    ratings = {}
    try:
        rating_divs = driver.find_elements(By.CSS_SELECTOR, '.choices[data-param="trating"] .ui_checkbox.item')
        for div in rating_divs:
            label = div.find_element(By.CSS_SELECTOR, '.row_label').text
            number_text = div.find_element(By.CSS_SELECTOR, '.row_num').text
            formatted_number_text = number_text.replace(",", "")
            number = int(formatted_number_text) if formatted_number_text else 0
            ratings[label] = number
        if "" in ratings:
            del ratings[""]
    except NoSuchElementException as e:
        logging.error(f"Could not extract ratings. Error: {str(e)}")
    return ratings

# Selection functions beneath this line 

# This function clicks on the most reviewed restaurant or Michelin restaurants if isMichelin
def click_on_most_reviewed(driver, isMichelin):
    wait_for_page_load(driver)
    restaurant_blocks = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.location-meta-block'))
    )
    restaurant_blocks = restaurant_blocks[:3]
    review_count_elements, restaurant_blocks = check_michelin(driver, restaurant_blocks, isMichelin)
    max_review_count = 0
    max_reviews_element = None

    for element in review_count_elements:
        review_texts = element.text.split()
        if review_texts is None or len(review_texts) == 0:
            continue
        formatted_review_text = review_texts[0].replace(",", "")
        try:
            reviews = int(formatted_review_text)
            if reviews > max_review_count:
                max_review_count = reviews
                max_reviews_element = element
        except ValueError:
            continue
        
    if max_reviews_element:
        driver.execute_script("arguments[0].click();", max_reviews_element)

# This function narrows down a search to remove/add Michelin restaurants
def check_michelin(driver, restaurant_blocks, isMichelin):
    filtered_blocks = []
    filtered_reviews = []
    for block in restaurant_blocks:
        try:
            michelin_title_element = WebDriverWait(block, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.michelin-title'))
            )
            michelin_title_text = michelin_title_element.text
        except TimeoutException:
            michelin_title_text = ""
        condition = (isMichelin and michelin_title_text.lower() == "michelin") or \
                    (not isMichelin and michelin_title_text.lower() != "michelin")
        
        if condition:
            filtered_blocks.append(block)
            review_count_element = block.find_element(By.CSS_SELECTOR, 'a.review_count')
            filtered_reviews.append(review_count_element)
    return filtered_reviews, filtered_blocks


# Function accepts cookies
def accept_cookies(driver):
    try:
        accept_cookies_button = WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        accept_cookies_button.click()
        time.sleep(7)
    except Exception as e:
        logging.info("Accept cookies pop-up not available")
        pass
    return

# This function clicks on the first element
def click_on_first_element(driver):
    top_result = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "result-title"))
    )   
    driver.execute_script("arguments[0].click();", top_result)

# This function avoids the sign in dialog for Google
def avoid_sign_in_dialog(driver):
    try:
        iframe = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[title='Sign in with Google Dialogue']"))
        )
        driver.switch_to.frame(iframe)
        close_sign_in_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "close"))
        )
        close_sign_in_button.click()
        driver.switch_to.default_content()
    except TimeoutException as e:
        logging.info("Timed out waiting for iframe or close button")
    except NoSuchElementException as e:
        logging.info("Could not find iframe or close button")
    except Exception as e:
        logging.info("An unknown error occurred while trying to close the sign in dialog")
    finally:
        wait_for_page_load(driver)  # Wait for the page to load after closing the dialog

# Function that determines whether the page has loaded
def wait_for_page_load(driver, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return document.readyState") == "complete")
    except TimeoutException:
        logging.info("Timed out waiting for page to load")

def attempt_global_search(driver):
    try:
        search_instead_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'p.original-query'))
        )
        driver.execute_script("arguments[0].click();", search_instead_button)
    except TimeoutException:
        logging.info("Search query does not contain global results")
        pass

# This function uses Selenium to scrape the data from TripAdvisor
def scrape_location_data(query, isMichelin):
    options = Options()
    options.set_preference("permissions.default.geo", 1)  

    driver = webdriver.Firefox(options=options)
    screen_width = driver.execute_script("return window.screen.width")
    screen_height = driver.execute_script("return window.screen.height")

    # Prevent dynamic rendering from causing issues 
    driver.set_window_size(screen_width // 2, screen_height)
    try:
        # Search for query
        driver.get("https://www.tripadvisor.com")
        logging.info(f"Searching for '{query}'")
        try:
            search_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'][title='Search']"))
            )
            search_box.send_keys(query)
            search_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button[type='submit'][formaction*='Search']"))
            )
            search_button.click()
        except TimeoutException:
            logging.info(f"Could not find search box or search button for '{query}'")
            pass

        # Dismiss location alert
        try:
            WebDriverWait(driver, 6).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            alert.accept()
        except Exception as e:
            logging.info("Location alert not present")
            pass

        # Accept cookies
        accept_cookies(driver)

        # Attempt to dismiss the sign in dialog
        avoid_sign_in_dialog(driver)

        # Wait for search to complete
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "search-results-list"))
        )

        # Change to global perspective
        attempt_global_search(driver)

        # Accept cookies
        accept_cookies(driver)

        # Call selection function here
        click_on_most_reviewed(driver, isMichelin)

        # Avoid sign in dialog
        avoid_sign_in_dialog(driver)

        # Wait for page to load
        wait_for_page_load(driver)

        # Obtain the data
        try:
            restaurant_name_element = find_element_with_retry(driver, [(By.CSS_SELECTOR, 'h1[data-test-target="top-info-header"]'), (By.CSS_SELECTOR, 'h1[data-automation="mainH1"]')])
            restaurant_name = restaurant_name_element.text
        except NoSuchElementException as e:
            logging.error(f"Could not find restaurant name. Error: {str(e)}")
            raise Exception("Failed to find restaurant name")

        try:
            address_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="https://maps.google.com/maps"] span'))
            )
            address = address_element.text
        except TimeoutException as e:
            logging.error(f"Could not find address. Error: {str(e)}")
            raise Exception("Failed to find address")

        url = driver.current_url

        try:
            website_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.LINK_TEXT, "Website"))
            )
            restaurant_url = website_element.get_attribute("href")
        except TimeoutException as e:
            logging.error(f"Could not find restaurant URL. Error: {str(e)}")
            restaurant_url = ""

        try:
            cuisines_label = driver.find_element(By.XPATH, "//div[text()='CUISINES']")
            cuisines_text = cuisines_label.find_element(By.XPATH, "following-sibling::div").text
        except NoSuchElementException as e:
            logging.error(f"Could not find cuisines. Error: {str(e)}")
            cuisines_text = ""

        try:
            reviews_element = driver.find_element(By.XPATH, "//a[@href='#REVIEWS']/span[contains(., 'reviews')]")
            reviews_text = reviews_element.text
            number_of_reviews = reviews_text.split()[0].replace(",", "")
        except NoSuchElementException as e:
            logging.error(f"Could not find number of reviews. Error: {str(e)}")
            number_of_reviews = "0"

        ratings = get_ratings(driver)

        debug_print(restaurant_name, address, url, restaurant_url, cuisines_text, number_of_reviews, ratings)
        write_to_csv(restaurant_name, address, url, restaurant_url, cuisines_text, number_of_reviews, ratings)

    finally:
        if driver:
            	driver.quit()

# This function is used to find an element with retries
def find_element_with_retry(driver, selectors, retries=3, delay=1):
    for i in range(retries):
        for by, value in selectors:
            try:
                return driver.find_element(by, value)
            except NoSuchElementException:
                continue
        if i < retries - 1:
            time.sleep(delay)
    raise NoSuchElementException(f"Could not find element with selectors: {selectors}")


# This function is used for debugging purposese to print out the data
def debug_print(restaurant_name, address, url, restaurant_url, cuisines_text, number_of_reviews, ratings):
    print(f"Restaurant Name: {restaurant_name}")
    print(f"Address: {address}")
    print(f"URL: {url}")
    print(f"Restaurant URL: {restaurant_url}")
    print(f"Cuisines: {cuisines_text}")
    print(f"Number of Reviews: {number_of_reviews}")
    print("Ratings Breakdown:")
    print(json.dumps(ratings, indent=4))

# This function is used to write the data to a CSV file
def write_to_csv(restaurant_name, address, url, restaurant_url, cuisines_text, number_of_reviews, ratings):
    write_header = not os.path.exists(output_csv) 
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Restaurant Name', 'Address', 'URL', 'Restaurant URL', 'Cuisines', 'Number of Reviews']
        fieldnames.extend([f"{key} reviews" for key in ratings.keys()])
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()

        writer.writerow({
            'Restaurant Name': restaurant_name,
            'Address': address,
            'URL': url,
            'Restaurant URL': restaurant_url,
            'Cuisines': cuisines_text,
            'Number of Reviews': number_of_reviews,
            'Excellent reviews': ratings['Excellent'],
            'Very good reviews': ratings['Very good'],
            'Average reviews': ratings['Average'],
            'Poor reviews': ratings['Poor'],
            'Terrible reviews': ratings['Terrible']
        })


# query = "212"
# scrape_location_data(query, True)
