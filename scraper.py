from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException

def scrape_location_data(query):
    driver = webdriver.Firefox()
    try:
        driver.get("https://www.tripadvisor.ie")
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'][title='Search']"))
        )
        search_box.send_keys(query)
        search_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button[type='submit'][formaction*='Search']"))
        )
        search_button.click()
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "search-results-list"))
        )
        accept_cookies_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
        )
        accept_cookies_button.click()
        top_result = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "result-title"))
        )   
        driver.execute_script("arguments[0].click();", top_result)

        try:
            iframe = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//iframe[@title='Sign in with Google Dialogue']"))
            )
            driver.switch_to.frame(iframe)
            print("Switched to iframe")
            close_sign_in_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "close"))
            )
            print("Found close button")
            driver.execute_script("arguments[0].scrollIntoView(true);", close_sign_in_button)
            driver.execute_script("arguments[0].click();", close_sign_in_button)
            driver.switch_to.default_content()
        except TimeoutException as e:
            print("Timed out waiting for iframe or close button")
            print(e)
        except NoSuchElementException as e:
            print("Element not found")
            print(e)
        except Exception as e:
            print("An unexpected error occurred")
            print(e)


    
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "taplc_top_info_0"))
        )
        address_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href="#MAPVIEW"]'))
        )
        address = address_element.text
        url = driver.current_url
        website_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Website"))
        )
        restaurant_url = website_element.get_attribute("href")

        print(f"Address: {address}")
        print(f"URL: {url}")
        print(f"Restaurant URL: {restaurant_url}")
    finally:
        driver.quit()

if __name__ == "__main__":
    query = "La Brasserie"
    scrape_location_data(query)
