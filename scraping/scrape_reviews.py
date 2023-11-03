import csv
from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementNotInteractableException

# This function scrapes the reviews for a particular Trip Advisor URL
def scrape_reviews(name, url, output_file, pages_to_scrape):
    def wait_for_page_load(driver, timeout=10):
        try:
            WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return document.readyState") == "complete")
        except TimeoutException:
            print("Error")

    def accept_cookies(driver):
        try:
            accept_cookies_button = WebDriverWait(driver, 6).until(
                EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler"))
            )
            accept_cookies_button.click()
        except Exception as e:
            print("Accept cookies pop-up not available")
            pass

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
            print("Timed out waiting for iframe or close button")
        except NoSuchElementException as e:
            print("Could not find iframe or close button")
        except Exception as e:
            print("An unknown error occurred while trying to close the sign in dialog")
        finally:
            wait_for_page_load(driver)

    driver = webdriver.Firefox()
    screen_width = driver.execute_script("return window.screen.width")
    screen_height = driver.execute_script("return window.screen.height")

    driver.set_window_size(screen_width // 2, screen_height)
    driver.get(url)
    csvFile = open(output_file, 'a', encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    for i in range(0, pages_to_scrape):
        wait_for_page_load(driver)
        if i == 0:
            accept_cookies(driver)
        time.sleep(1)
        wait_for_page_load(driver)
        if i == 0:
            avoid_sign_in_dialog(driver)
                
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, ".taLnk.ulBlueLinks[onclick*='clickExpand']")
            for element in elements:
                try:
                    driver.execute_script("arguments[0].scrollIntoView();", element)
                    driver.execute_script("arguments[0].click();", element)
                except StaleElementReferenceException:
                    print("StaleElementReferenceException: retrying...")
                    continue  # retry the loop
        except NoSuchElementException:
            print("No 'View more' button found")
            pass



        wait_for_page_load(driver)
        try:
            view_more_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '.see-more-mobile.ui_button.primary'))
            )
            driver.execute_script("arguments[0].click();", view_more_button)
        except (TimeoutException, NoSuchElementException):
            print("Timed out waiting for 'View more reviews' button")
            pass
        container = driver.find_elements(By.XPATH, "//div[@class='ui_column is-9']")

        for j in range(len(container)):
            try:
                rating_element = WebDriverWait(container[j], 10).until(
                    EC.presence_of_element_located((By.XPATH, ".//span[contains(@class, 'ui_bubble_rating bubble_')]"))
                )
                rating = rating_element.get_attribute("class").split("_")[3]
            except TimeoutException:
                print("Timed out waiting for rating element")
                break
            except StaleElementReferenceException:
                print("StaleElementReferenceException: retrying...")
                continue  # retry the loop
            try:
                title_element = WebDriverWait(container[j], 10).until(
                    EC.presence_of_element_located((By.XPATH, ".//span[@class='noQuotes']"))
                )
                title = title_element.text
            except TimeoutException:
                print("Timed out waiting for title element")
                break
            except StaleElementReferenceException:
                print("StaleElementReferenceException: retrying...")
                continue  # retry the loop
            try:
                review_element = WebDriverWait(container[j], 10).until(
                    EC.presence_of_element_located((By.XPATH, ".//p[@class='partial_entry']"))
                )
                review = review_element.text.replace("\n", "  ")
            except TimeoutException:
                print("Timed out waiting for review element")
                break
            except StaleElementReferenceException:
                print("StaleElementReferenceException: retrying...")
                continue  # retry the loop
            print(f"rating: {rating}")
            print(f"title: {title}")
            print(f"review: {review}")
            csvWriter.writerow([name, rating, title, review])


        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a.nav.next.ui_button.primary'))
            )
            driver.execute_script("arguments[0].scrollIntoView();", next_button)
            driver.execute_script("arguments[0].click();", next_button)
        except TimeoutException:
            print("Timed out waiting for next button to be clickable")
            break
        except ElementNotInteractableException:
            print("Next button is not interactable")
            break
    driver.quit()

    

