---
layout: post
title: Scrapping App Reviews for popular roboadvisors in Singapore using Python
description: Using linear regression to predict housing prices in Melbourne. # Add post description (optional)
img: '/app-review-scrap/scrapappreviews.jpeg' # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Data Analytics, Linear Regression]
---

Using python to scrapping app reviews for Syfe, Endowus and Stashaway from Google Play and Apple Store.

Part 1: Scrap reviews from Apple Store  
Part 2: Scrap reviews from Google Play Store  
Part 3: Combining both reviews

Reference: 
- Apple Store Scrapper: https://python.plainenglish.io/scraping-app-store-reviews-with-python-90e4117ccdfb
- Google Play Store Scraper: https://python.plainenglish.io/scraping-storing-google-play-app-reviews-with-python-5640c933c476

```python
## Importing relevant libraries
import pandas as pd
import numpy as np

# for scraping app info from App Store
from itunes_app_scraper.scraper import AppStoreScraper

# for scraping app reviews from App Store
from app_store_scraper import AppStore

# for pretty printing data structures
from pprint import pprint

# for keeping track of timing
import datetime as dt
from tzlocal import get_localzone

# for building in wait times
import random
import time
```

    /Users/a844133yara.com/.pyenv/versions/3.9.5/envs/python_playground/lib/python3.9/site-packages/pandas/compat/__init__.py:124: UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.
      warnings.warn(msg)


# Part 1 - Scrap reviews from Apple Store


```python
## Read in file containing app names and IDs
apple_app_df = pd.read_excel('app_info.xlsx', sheet_name='apple')
print(f"""
Printing first few rows of app's info in the csv file:
------------------------------------------------------
{apple_app_df.head()}
""")

## Get list of app names and app IDs
apple_app_names = list(apple_app_df['app_name'])
apple_app_ids = list(apple_app_df['iOS_app_id'])
```

    
    Printing first few rows of app's info in the csv file:
    ------------------------------------------------------
        app_name                 iOS_app_name  iOS_app_id
    0       Syfe           syfe-invest-better  1497156434
    1    Endowus  endowus-invest-cpf-srs-cash  1531067679
    2  StashAway    stashaway-invest-and-save  1229966330
    



```python
## Set up App Store Scraper
scraper = AppStoreScraper()
apple_app_store_list = list(scraper.get_multiple_app_details(apple_app_ids))

## Pretty print the data for the first app
pprint(apple_app_store_list[0])
```

    https://itunes.apple.com/lookup?id=1497156434&country=nl&entity=software
    https://itunes.apple.com/lookup?id=1531067679&country=nl&entity=software
    https://itunes.apple.com/lookup?id=1229966330&country=nl&entity=software
    {'advisories': '',
     'appletvScreenshotUrls': '',
     'artistId': 1497156433,
     'artistName': 'Syfe Pte. Ltd.',
     'artistViewUrl': 'https://apps.apple.com/nl/developer/syfe-pte-ltd/id1497156433?uo=4',
     'artworkUrl100': 'https://is4-ssl.mzstatic.com/image/thumb/Purple115/v4/6b/56/2e/6b562e57-e6e9-6354-1a64-632570e25c9b/source/100x100bb.jpg',
     'artworkUrl512': 'https://is4-ssl.mzstatic.com/image/thumb/Purple115/v4/6b/56/2e/6b562e57-e6e9-6354-1a64-632570e25c9b/source/512x512bb.jpg',
     'artworkUrl60': 'https://is4-ssl.mzstatic.com/image/thumb/Purple115/v4/6b/56/2e/6b562e57-e6e9-6354-1a64-632570e25c9b/source/60x60bb.jpg',
     'averageUserRating': 0,
     'averageUserRatingForCurrentVersion': 0,
     'bundleId': 'com.syfe',
     'contentAdvisoryRating': '4+',
     'currency': 'EUR',
     'currentVersionReleaseDate': '2021-09-01T13:39:01Z',
     'description': 'Whether you plan to buy a house, save up for your children’s '
                    'education, retire or simply grow your wealth, we have '
                    'portfolios that caters to your different financial goals:\n'
                    '\n'
                    '1. Syfe Core: All-in-one portfolios diversified across top '
                    'global stock, bond and gold ETFs that track popular indexes '
                    'such as S&P 500, NASDAQ and more\n'
                    '\n'
                    '- Core Equity100: for those seeking 100% exposure to global '
                    'equities and comfortable with taking higher risk for '
                    'potential higher long-term returns\n'
                    '- Core Growth: for those seeking higher return potential and '
                    'long-term growth\n'
                    '- Core Balanced: for those seeking income and long-term '
                    'growth\n'
                    '- Core Defensive: for those seeking stable returns and '
                    'long-term income\n'
                    '\n'
                    '2. Syfe REIT+: Passive Income Portfolio\n'
                    '\n'
                    '- REIT+: for those seeking to invest in Singapore’s largest '
                    'real estate investment trusts (REITs) such as Ascendas REITs, '
                    'CapitaLand Integrated Commercial Trust, Mapletree Commercial '
                    'Trust, and more.\n'
                    '\n'
                    '3. Syfe Cash+: Cash Management Portfolio\n'
                    '\n'
                    '- Cash+: for those seeking higher returns on their savings '
                    'than at banks with a 1.5% p.a. projected return\n'
                    '\n'
                    'OUR APP FEATURES\n'
                    'Investing should be easy and quick. With the Syfe app, you '
                    'can enjoy seamless investing and the benefits of:\n'
                    '\n'
                    '- Investing in minutes. Easily add however many portfolios '
                    'you want and conveniently transfer your funds between '
                    'portfolios\n'
                    '- Monitoring your investments at a glance. See how well your '
                    'investments are performing, your returns and easily track the '
                    'dividends you have received\n'
                    '- Making it your own. Get access to over 100 curated ETFs and '
                    'Funds and customise your own asset allocations to express '
                    'your own investment convictions. \n'
                    '- Learning as you go. Explore our collection of research '
                    'guides, expert advice commentary, webinars and magazines – '
                    'filled with the latest financial news\n'
                    '- Scheduling calls for personalised wealth guidance. Access '
                    'our team of wealth experts, who are dedicated to ensuring you '
                    'make smarter decisions about your personal finances\n'
                    '\n'
                    'WHY INVESTORS CHOOSE SYFE\n'
                    'Join an ever growing number of investors that invest with '
                    'Syfe and take advantage of our offerings now:\n'
                    '\n'
                    '- Simple, transparent pricing. Enjoy no brokerage fees, no '
                    'entry fees, no withdrawal fees and no hidden charges. We '
                    'charge one simple fee: an all-inclusive annual fee of 0.4% to '
                    '0.65% on your invested capital. That’s all.\n'
                    '- Access a wide range of asset classes, countries and '
                    'sectors. Ranging from portfolios that invest in the '
                    'fastest-growing economies such as China to leading companies '
                    'such as Google, Microsoft, Tencent and more, we have all your '
                    'bases covered.\n'
                    '- Great for beginners, experts and everyone in between. Even '
                    'if you’re not sure how to invest, we’ll help you realise your '
                    'financial goals with our risk questionnaire and recommend the '
                    'portfolio that is best suited for your goals.\n'
                    '\n'
                    'AS FEATURED BY\n'
                    'Hear from industry leaders:\n'
                    '\n'
                    '“Better than DIY” - MoneySmart\n'
                    '“12 Investment Apps to Multiply Your Savings” - '
                    'TheSmartLocal\n'
                    '“Good for dollar cost averaging and accumulation” - '
                    'InvestmentMoats\n'
                    '\n'
                    'REACH OUT TO US\n'
                    'Please contact us if you have any queries via:\n'
                    '\n'
                    '- Our Live Chat service on the Syfe app and web dashboard\n'
                    '- Email at support@syfe.com\n'
                    '- Call us at +65 3138 1215 between 9:00 am and 6:00 pm Monday '
                    'through Friday\n'
                    '\n'
                    'MAS Capital Markets Services (CMS) License - CMS100837. '
                    'Located at 4 Robinson Rd, #11-01 The House Of Eden, Singapore '
                    '048543.',
     'features': '',
     'fileSizeBytes': '45821952',
     'formattedPrice': 'Gratis',
     'genreIds': '6015',
     'genres': 'Financiën',
     'ipadScreenshotUrls': '',
     'isGameCenterEnabled': False,
     'isVppDeviceBasedLicensingEnabled': True,
     'kind': 'software',
     'languageCodesISO2A': 'EN',
     'minimumOsVersion': '11.0',
     'price': 0.0,
     'primaryGenreId': 6015,
     'primaryGenreName': 'Finance',
     'releaseDate': '2020-02-24T08:00:00Z',
     'releaseNotes': '1. See statuses for various transactions - withdrawals, fund '
                     'transfers and new funds added\n'
                     '2. Download new and past monthly statements from account '
                     'settings\n'
                     '3. See more detailed information on portfolios when adding '
                     'new portfolios\n'
                     '4. Minor fixes and performance improvements',
     'screenshotUrls': 'https://is5-ssl.mzstatic.com/image/thumb/PurpleSource125/v4/0a/6a/87/0a6a8739-4307-6a2c-51d7-779702ca47d8/5fefcd46-1669-4c46-a1b6-c00f72443d37_card_1.jpg/392x696bb.jpg,https://is4-ssl.mzstatic.com/image/thumb/PurpleSource115/v4/83/0a/75/830a756b-eef4-bcb4-64cc-114e256664c5/f67a8afd-c5b3-41ed-ba2f-f68eda51fbaf_card_2.jpg/392x696bb.jpg,https://is5-ssl.mzstatic.com/image/thumb/PurpleSource115/v4/3f/87/75/3f877564-f390-df52-b2cf-35b5b9e39d60/520391b1-45d1-4237-91cb-979a953ea3f8_card_3.jpg/392x696bb.jpg,https://is1-ssl.mzstatic.com/image/thumb/PurpleSource125/v4/1b/9c/23/1b9c23bb-789b-6e9b-b49d-79e699bb729a/d8186c68-20af-40fa-8ac7-d8f62a526e07_card_4.jpg/392x696bb.jpg,https://is5-ssl.mzstatic.com/image/thumb/PurpleSource125/v4/4d/43/e1/4d43e185-bf4b-9cb9-9c35-a47cf194ad33/3e7009c4-bf73-49e0-aca5-ea5baf856a64_card_5.jpg/392x696bb.jpg,https://is4-ssl.mzstatic.com/image/thumb/PurpleSource125/v4/95/a2/ac/95a2ac4e-2641-0c4d-c435-f86f563d26fe/f06d1673-5240-4b6e-a29b-e7d21be00eba_card_6.jpg/392x696bb.jpg,https://is4-ssl.mzstatic.com/image/thumb/PurpleSource125/v4/81/ec/5b/81ec5bbf-3bae-5a5c-10da-44a985f1e14e/b3d6acd1-074d-4c01-bf89-1440c7f5cd48_card_7.jpg/392x696bb.jpg,https://is1-ssl.mzstatic.com/image/thumb/PurpleSource125/v4/b9/c4/4a/b9c44abc-dd88-6192-0a24-72667887f3ab/532d64a2-f6c3-4e7a-8b4b-6028664e92c1_card_8.jpg/392x696bb.jpg',
     'sellerName': 'Svava Pte Ltd',
     'sellerUrl': 'https://www.syfe.com',
     'supportedDevices': 'iPhone5s-iPhone5s,iPadAir-iPadAir,iPadAirCellular-iPadAirCellular,iPadMiniRetina-iPadMiniRetina,iPadMiniRetinaCellular-iPadMiniRetinaCellular,iPhone6-iPhone6,iPhone6Plus-iPhone6Plus,iPadAir2-iPadAir2,iPadAir2Cellular-iPadAir2Cellular,iPadMini3-iPadMini3,iPadMini3Cellular-iPadMini3Cellular,iPodTouchSixthGen-iPodTouchSixthGen,iPhone6s-iPhone6s,iPhone6sPlus-iPhone6sPlus,iPadMini4-iPadMini4,iPadMini4Cellular-iPadMini4Cellular,iPadPro-iPadPro,iPadProCellular-iPadProCellular,iPadPro97-iPadPro97,iPadPro97Cellular-iPadPro97Cellular,iPhoneSE-iPhoneSE,iPhone7-iPhone7,iPhone7Plus-iPhone7Plus,iPad611-iPad611,iPad612-iPad612,iPad71-iPad71,iPad72-iPad72,iPad73-iPad73,iPad74-iPad74,iPhone8-iPhone8,iPhone8Plus-iPhone8Plus,iPhoneX-iPhoneX,iPad75-iPad75,iPad76-iPad76,iPhoneXS-iPhoneXS,iPhoneXSMax-iPhoneXSMax,iPhoneXR-iPhoneXR,iPad812-iPad812,iPad834-iPad834,iPad856-iPad856,iPad878-iPad878,iPadMini5-iPadMini5,iPadMini5Cellular-iPadMini5Cellular,iPadAir3-iPadAir3,iPadAir3Cellular-iPadAir3Cellular,iPodTouchSeventhGen-iPodTouchSeventhGen,iPhone11-iPhone11,iPhone11Pro-iPhone11Pro,iPadSeventhGen-iPadSeventhGen,iPadSeventhGenCellular-iPadSeventhGenCellular,iPhone11ProMax-iPhone11ProMax,iPhoneSESecondGen-iPhoneSESecondGen,iPadProSecondGen-iPadProSecondGen,iPadProSecondGenCellular-iPadProSecondGenCellular,iPadProFourthGen-iPadProFourthGen,iPadProFourthGenCellular-iPadProFourthGenCellular,iPhone12Mini-iPhone12Mini,iPhone12-iPhone12,iPhone12Pro-iPhone12Pro,iPhone12ProMax-iPhone12ProMax,iPadAir4-iPadAir4,iPadAir4Cellular-iPadAir4Cellular,iPadEighthGen-iPadEighthGen,iPadEighthGenCellular-iPadEighthGenCellular,iPadProThirdGen-iPadProThirdGen,iPadProThirdGenCellular-iPadProThirdGenCellular,iPadProFifthGen-iPadProFifthGen,iPadProFifthGenCellular-iPadProFifthGenCellular',
     'trackCensoredName': 'Syfe: Invest Better',
     'trackContentRating': '4+',
     'trackId': 1497156434,
     'trackName': 'Syfe: Invest Better',
     'trackViewUrl': 'https://apps.apple.com/nl/app/syfe-invest-better/id1497156434?uo=4',
     'userRatingCount': 0,
     'userRatingCountForCurrentVersion': 0,
     'version': '4.80',
     'wrapperType': 'software'}



```python
# Converting list into dataframe
apple_app_info_df = pd.DataFrame(apple_app_store_list)
apple_app_info_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artworkUrl60</th>
      <th>artworkUrl100</th>
      <th>artistViewUrl</th>
      <th>screenshotUrls</th>
      <th>ipadScreenshotUrls</th>
      <th>artworkUrl512</th>
      <th>appletvScreenshotUrls</th>
      <th>supportedDevices</th>
      <th>advisories</th>
      <th>isGameCenterEnabled</th>
      <th>...</th>
      <th>currency</th>
      <th>description</th>
      <th>artistId</th>
      <th>artistName</th>
      <th>genres</th>
      <th>price</th>
      <th>bundleId</th>
      <th>version</th>
      <th>wrapperType</th>
      <th>userRatingCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://apps.apple.com/nl/developer/syfe-pte-l...</td>
      <td>https://is5-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td></td>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td></td>
      <td>iPhone5s-iPhone5s,iPadAir-iPadAir,iPadAirCellu...</td>
      <td></td>
      <td>False</td>
      <td>...</td>
      <td>EUR</td>
      <td>Whether you plan to buy a house, save up for y...</td>
      <td>1497156433</td>
      <td>Syfe Pte. Ltd.</td>
      <td>Financiën</td>
      <td>0.0</td>
      <td>com.syfe</td>
      <td>4.80</td>
      <td>software</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://apps.apple.com/nl/developer/endowus/id...</td>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td></td>
      <td>https://is4-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td></td>
      <td>iPhone5s-iPhone5s,iPadAir-iPadAir,iPadAirCellu...</td>
      <td></td>
      <td>False</td>
      <td>...</td>
      <td>EUR</td>
      <td>Thousands invest and grow their Cash, CPF &amp; SR...</td>
      <td>1531067681</td>
      <td>Endowus</td>
      <td>Financiën,Lifestyle</td>
      <td>0.0</td>
      <td>com.endowus.mobileapp</td>
      <td>1.3.1</td>
      <td>software</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://is1-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://is1-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://apps.apple.com/nl/developer/asia-wealt...</td>
      <td>https://is2-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://is1-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td>https://is1-ssl.mzstatic.com/image/thumb/Purpl...</td>
      <td></td>
      <td>iPhone5s-iPhone5s,iPadAir-iPadAir,iPadAirCellu...</td>
      <td></td>
      <td>False</td>
      <td>...</td>
      <td>EUR</td>
      <td>StashAway is where intelligent investing meets...</td>
      <td>1229625364</td>
      <td>Asia Wealth Platform Pte Ltd</td>
      <td>Financiën</td>
      <td>0.0</td>
      <td>com.awp.stashaway</td>
      <td>11.95.2</td>
      <td>software</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 44 columns</p>
</div>



Given that there are no user rating counts, we can ignore itunes ratings in our analysis. 

## Scrapping App Reviews from Apple Store


```python
# Empty list for storing reviews
apple_app_reviews = []

## Set up loop to go through all apps
for app_name, app_id in zip(apple_app_names, apple_app_ids):
    
    # Get start time
    start = dt.datetime.now(tz=get_localzone())
    fmt= "%m/%d/%y - %T %p"
    
    # Print starting output for app
    print('---'*20)
    print('---'*20)    
    print(f'***** {app_name} started at {start.strftime(fmt)}')
    print()
    
    # Instantiate AppStore for app
    app_ = AppStore(country='sg', app_name=app_name, app_id=app_id)
    
    # Scrape reviews posted since February 28, 2020 and limit to 10,000 reviews
    app_.review(how_many=10000,
                after=dt.datetime(2020, 2, 28),
                sleep=random.randint(20,25))
    
    reviews = app_.reviews
    
    # Add keys to store information about which app each review is for
    for rvw in reviews:
        rvw['app_name'] = app_name
        rvw['app_id'] = app_id
    
    # Print update that scraping was completed
    print(f"""Done scraping {app_name}. 
    Scraped a total of {app_.reviews_count} reviews.\n""")

     # Convert list of dicts to Pandas DataFrame
    review_df = pd.DataFrame(reviews)
    apple_app_reviews.append(review_df)
    
    # Get end time
    end = dt.datetime.now(tz=get_localzone())
    
    # Print ending output for app
    print(f"""Successfully wrote {app_name} reviews to df
    at {end.strftime(fmt)}.\n""")
    print(f'Time elapsed for {app_name}: {end-start}')
    print('---'*20)
    print('---'*20)
    print('\n')
    
    # Wait 5 to 10 seconds to start scraping next app
    time.sleep(random.randint(5,10))
```

    ------------------------------------------------------------
    ------------------------------------------------------------
    ***** Syfe started at 09/04/21 - 23:19:19 PM
    


    2021-09-04 23:19:21,123 [INFO] Base - Initialised: AppStore('sg', 'syfe', 1497156434)
    2021-09-04 23:19:21,124 [INFO] Base - Ready to fetch reviews from: https://apps.apple.com/sg/app/syfe/id1497156434
    2021-09-04 23:19:41,173 [INFO] Base - [id:1497156434] Fetched 20 reviews (20 fetched in total)
    2021-09-04 23:20:21,540 [INFO] Base - [id:1497156434] Fetched 57 reviews (57 fetched in total)
    2021-09-04 23:20:21,834 [INFO] Base - [id:1497156434] Fetched 64 reviews (64 fetched in total)


    Done scraping Syfe. 
        Scraped a total of 64 reviews.
    
    Successfully wrote Syfe reviews to df
        at 09/04/21 - 23:20:21 PM.
    
    Time elapsed for Syfe: 0:01:02.025785
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    
    ------------------------------------------------------------
    ------------------------------------------------------------
    ***** Endowus started at 09/04/21 - 23:20:29 PM
    


    2021-09-04 23:20:31,468 [INFO] Base - Initialised: AppStore('sg', 'endowus', 1531067679)
    2021-09-04 23:20:31,470 [INFO] Base - Ready to fetch reviews from: https://apps.apple.com/sg/app/endowus/id1531067679
    2021-09-04 23:20:54,512 [INFO] Base - [id:1531067679] Fetched 20 reviews (20 fetched in total)
    2021-09-04 23:21:41,308 [INFO] Base - [id:1531067679] Fetched 60 reviews (60 fetched in total)
    2021-09-04 23:21:41,712 [INFO] Base - [id:1531067679] Fetched 70 reviews (70 fetched in total)


    Done scraping Endowus. 
        Scraped a total of 70 reviews.
    
    Successfully wrote Endowus reviews to df
        at 09/04/21 - 23:21:41 PM.
    
    Time elapsed for Endowus: 0:01:11.872447
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    
    ------------------------------------------------------------
    ------------------------------------------------------------
    ***** StashAway started at 09/04/21 - 23:21:48 PM
    


    2021-09-04 23:21:50,044 [INFO] Base - Initialised: AppStore('sg', 'stashaway', 1229966330)
    2021-09-04 23:21:50,045 [INFO] Base - Ready to fetch reviews from: https://apps.apple.com/sg/app/stashaway/id1229966330
    2021-09-04 23:22:12,087 [INFO] Base - [id:1229966330] Fetched 17 reviews (17 fetched in total)
    2021-09-04 23:22:56,189 [INFO] Base - [id:1229966330] Fetched 50 reviews (50 fetched in total)
    2021-09-04 23:23:40,312 [INFO] Base - [id:1229966330] Fetched 83 reviews (83 fetched in total)
    2021-09-04 23:24:24,429 [INFO] Base - [id:1229966330] Fetched 119 reviews (119 fetched in total)
    2021-09-04 23:25:08,547 [INFO] Base - [id:1229966330] Fetched 146 reviews (146 fetched in total)
    2021-09-04 23:25:52,710 [INFO] Base - [id:1229966330] Fetched 180 reviews (180 fetched in total)
    2021-09-04 23:26:36,814 [INFO] Base - [id:1229966330] Fetched 213 reviews (213 fetched in total)
    2021-09-04 23:27:21,477 [INFO] Base - [id:1229966330] Fetched 244 reviews (244 fetched in total)
    2021-09-04 23:28:06,102 [INFO] Base - [id:1229966330] Fetched 273 reviews (273 fetched in total)
    2021-09-04 23:28:50,685 [INFO] Base - [id:1229966330] Fetched 309 reviews (309 fetched in total)
    2021-09-04 23:29:35,230 [INFO] Base - [id:1229966330] Fetched 337 reviews (337 fetched in total)
    2021-09-04 23:30:19,753 [INFO] Base - [id:1229966330] Fetched 366 reviews (366 fetched in total)
    2021-09-04 23:31:04,274 [INFO] Base - [id:1229966330] Fetched 390 reviews (390 fetched in total)
    2021-09-04 23:31:04,524 [INFO] Base - [id:1229966330] Fetched 390 reviews (390 fetched in total)


    Done scraping StashAway. 
        Scraped a total of 390 reviews.
    
    Successfully wrote StashAway reviews to df
        at 09/04/21 - 23:31:04 PM.
    
    Time elapsed for StashAway: 0:09:15.818408
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    



```python
 # Convert list of dfs to Pandas DataFrame and write to csv
apple_reviews = pd.concat(apple_app_reviews)
```

# Part 2 - Scrap reviews from Google Play Store


```python
## Extracting data and relevant app names + Ids
google_app_df = pd.read_excel('app_info.xlsx',sheet_name='google')
print(google_app_df.head())

## Get list of app names and app IDs
google_app_names = list(google_app_df['app_name'])
google_app_ids = list(google_app_df['app_id'])
```

        app_name                 app_id
    0       Syfe               com.syfe
    1    Endowus  com.endowus.mobileapp
    2  StashAway      com.awp.stashaway



```python
## Loop through app IDs to get app info
google_app_info = []
for i in google_app_ids:
    info = app(i)
    del info['comments']
    google_app_info.append(info)

## Pretty print the data for the first app
pprint(google_app_info[0])

google_app_infos = pd.DataFrame(google_app_info)
# app_infos_df.to_csv('apps.csv', index=None, header=True)
google_app_infos
```

    {'adSupported': None,
     'androidVersion': '5.0',
     'androidVersionText': '5.0 and up',
     'appId': 'com.syfe',
     'containsAds': False,
     'contentRating': 'Everyone',
     'contentRatingDescription': None,
     'currency': 'USD',
     'description': 'Invest better, faster and smarter with Syfe. Syfe is a '
                    'digital wealth manager for investors who expect more – '
                    'greater transparency, smart\xader personalised portfolios, '
                    'and better investment outcomes.\r\n'
                    '\r\n'
                    'As one of Singapore’s fastest-growing robo advisors, we do '
                    'all the heavy lifting for you, from fund selection, '
                    'reinvesting dividends to rebalancing your portfolios and '
                    'more! All you have to do is sit back and watch your money '
                    'grow. Our digital investment advisor platform offers a '
                    "smarter way to invest with low fees and no minimums. We're "
                    'also licensed by the MAS, rest assured your money is safe '
                    'with us!\r\n'
                    '\r\n'
                    '\r\n'
                    'PORTFOLIOS FOR EVERYONE\r\n'
                    'Whether you plan to buy a house, save up for your children’s '
                    'education, retire or simply grow your wealth, we have '
                    'portfolios that caters to your different financial goals:\r\n'
                    '\r\n'
                    '1. Syfe Core: All-in-one portfolios diversified across top '
                    'global stock, bond and gold ETFs that track popular indexes '
                    'such as S&P 500, NASDAQ and more\r\n'
                    '\r\n'
                    '- Core Growth: for those seeking higher return potential and '
                    'long-term growth\r\n'
                    '- Core Balanced: for those seeking income and long-term '
                    'growth\r\n'
                    '- Core Defensive: for those seeking stable returns and '
                    'long-term income\r\n'
                    '- Global ARI: for those seeking to diversify with 11 risk '
                    'level options to choose from\r\n'
                    '\r\n'
                    '2. Syfe REIT+: Passive Income Portfolio\r\n'
                    '\r\n'
                    '- REIT+: For those seeking to invest in Singapore’s largest '
                    'real estate investment trusts (REITs) such as Ascendas REITs, '
                    'CapitaLand Integrated Commercial Trust, Mapletree Commercial '
                    'Trust, and more.\r\n'
                    '\r\n'
                    '3. Syfe Equity100: Pure Equity Portfolio \r\n'
                    '\r\n'
                    '- Equity100: For those seeking 100% exposure to global '
                    'equities. Gain access to 1,500+ stocks in the world’s best '
                    'companies from leading sectors such as technology and '
                    'healthcare\r\n'
                    '\r\n'
                    '4. Syfe Cash+: Cash Management Portfolio\r\n'
                    '\r\n'
                    '- Cash+: for those seeking higher returns on their savings '
                    'than at banks with a 1.75% p.a. projected return\r\n'
                    '\r\n'
                    'OUR APP FEATURES\r\n'
                    'Investing should be easy and quick. With the Syfe app, you '
                    'can enjoy seamless investing and the benefits of:\r\n'
                    '\r\n'
                    '- Investing in minutes. Easily add however many portfolios '
                    'you want and conveniently transfer your funds between '
                    'portfolios\r\n'
                    '- Monitoring your investments at a glance. See how well your '
                    'investments are performing, your returns and easily track the '
                    'dividends you have received\r\n'
                    '- Learning as you go. Explore our collection of research '
                    'guides, expert advice commentary, webinars and magazines – '
                    'filled with the latest financial news\r\n'
                    '- Scheduling calls for personalised wealth guidance. Access '
                    'our team of wealth experts, who are dedicated to ensuring you '
                    'make smarter decisions about your personal finances\r\n'
                    '\r\n'
                    'WHY INVESTORS CHOOSE SYFE\r\n'
                    'Join an evergrowing number of investors that invest with Syfe '
                    'and take advantage of our offerings now:\r\n'
                    '\r\n'
                    '- Simple, transparent pricing. Enjoy no brokerage fees, no '
                    'entry fees, no withdrawal fees and no hidden charges. We '
                    'charge one simple fee: an all-inclusive annual fee of 0.4% to '
                    '0.65% on your invested capital. That’s all.\r\n'
                    '- Access a wide range of asset classes, countries and '
                    'sectors. Ranging from portfolios that invest in the '
                    'fastest-growing economies such as China to leading companies '
                    'such as Google, Microsoft, Tencent and more, we have all your '
                    'bases covered.\r\n'
                    '- Great for beginners, experts and everyone in between. Even '
                    'if you’re not sure how to invest, we’ll help you realise your '
                    'financial goals with our risk questionnaire and recommend the '
                    'portfolio that is best suited for your goals.\r\n'
                    '\r\n'
                    'AS FEATURED BY\r\n'
                    'Hear from industry leaders:\r\n'
                    '\r\n'
                    '“Better than DIY” - MoneySmart\r\n'
                    '“12 Investment Apps to Multiply Your Savings” - '
                    'TheSmartLocal\r\n'
                    '“Good for dollar cost averaging and accumulation” - '
                    'InvestmentMoats\r\n'
                    '\r\n'
                    'REACH OUT TO US\r\n'
                    'Please contact us if you have any queries via:\r\n'
                    '\r\n'
                    '- Our Live Chat service on the Syfe app and web dashboard\r\n'
                    '- Email at support@syfe.com\r\n'
                    '- Call us at +65 3138 1215 between 9:00 am and 6:00 pm Monday '
                    'through Friday\r\n'
                    '\r\n'
                    '\r\n'
                    'MAS Capital Markets Services (CMS) License - CMS100837. '
                    'Located at 4 Robinson Rd, #11-01 The House Of Eden, Singapore '
                    '048543.',
     'descriptionHTML': 'Invest better, faster and smarter with Syfe. Syfe is a '
                        'digital wealth manager for investors who expect more – '
                        'greater transparency, smart\xader personalised '
                        'portfolios, and better investment outcomes.<br><br>As one '
                        'of Singapore’s fastest-growing robo advisors, we do all '
                        'the heavy lifting for you, from fund selection, '
                        'reinvesting dividends to rebalancing your portfolios and '
                        'more! All you have to do is sit back and watch your money '
                        'grow. Our digital investment advisor platform offers a '
                        'smarter way to invest with low fees and no minimums. '
                        'We&#39;re also licensed by the MAS, rest assured your '
                        'money is safe with us!<br><br><br>PORTFOLIOS FOR '
                        'EVERYONE<br>Whether you plan to buy a house, save up for '
                        'your children’s education, retire or simply grow your '
                        'wealth, we have portfolios that caters to your different '
                        'financial goals:<br><br>1. Syfe Core: All-in-one '
                        'portfolios diversified across top global stock, bond and '
                        'gold ETFs that track popular indexes such as S&amp;P 500, '
                        'NASDAQ and more<br><br>- Core Growth: for those seeking '
                        'higher return potential and long-term growth<br>- Core '
                        'Balanced: for those seeking income and long-term '
                        'growth<br>- Core Defensive: for those seeking stable '
                        'returns and long-term income<br>- Global ARI: for those '
                        'seeking to diversify with 11 risk level options to choose '
                        'from<br><br>2. Syfe REIT+: Passive Income '
                        'Portfolio<br><br>- REIT+: For those seeking to invest in '
                        'Singapore’s largest real estate investment trusts (REITs) '
                        'such as Ascendas REITs, CapitaLand Integrated Commercial '
                        'Trust, Mapletree Commercial Trust, and more.<br><br>3. '
                        'Syfe Equity100: Pure Equity Portfolio <br><br>- '
                        'Equity100: For those seeking 100% exposure to global '
                        'equities. Gain access to 1,500+ stocks in the world’s '
                        'best companies from leading sectors such as technology '
                        'and healthcare<br><br>4. Syfe Cash+: Cash Management '
                        'Portfolio<br><br>- Cash+: for those seeking higher '
                        'returns on their savings than at banks with a 1.75% p.a. '
                        'projected return<br><br>OUR APP FEATURES<br>Investing '
                        'should be easy and quick. With the Syfe app, you can '
                        'enjoy seamless investing and the benefits of:<br><br>- '
                        'Investing in minutes. Easily add however many portfolios '
                        'you want and conveniently transfer your funds between '
                        'portfolios<br>- Monitoring your investments at a glance. '
                        'See how well your investments are performing, your '
                        'returns and easily track the dividends you have '
                        'received<br>- Learning as you go. Explore our collection '
                        'of research guides, expert advice commentary, webinars '
                        'and magazines – filled with the latest financial '
                        'news<br>- Scheduling calls for personalised wealth '
                        'guidance. Access our team of wealth experts, who are '
                        'dedicated to ensuring you make smarter decisions about '
                        'your personal finances<br><br>WHY INVESTORS CHOOSE '
                        'SYFE<br>Join an evergrowing number of investors that '
                        'invest with Syfe and take advantage of our offerings '
                        'now:<br><br>- Simple, transparent pricing. Enjoy no '
                        'brokerage fees, no entry fees, no withdrawal fees and no '
                        'hidden charges. We charge one simple fee: an '
                        'all-inclusive annual fee of 0.4% to 0.65% on your '
                        'invested capital. That’s all.<br>- Access a wide range of '
                        'asset classes, countries and sectors. Ranging from '
                        'portfolios that invest in the fastest-growing economies '
                        'such as China to leading companies such as Google, '
                        'Microsoft, Tencent and more, we have all your bases '
                        'covered.<br>- Great for beginners, experts and everyone '
                        'in between. Even if you’re not sure how to invest, we’ll '
                        'help you realise your financial goals with our risk '
                        'questionnaire and recommend the portfolio that is best '
                        'suited for your goals.<br><br>AS FEATURED BY<br>Hear from '
                        'industry leaders:<br><br>“Better than DIY” - '
                        'MoneySmart<br>“12 Investment Apps to Multiply Your '
                        'Savings” - TheSmartLocal<br>“Good for dollar cost '
                        'averaging and accumulation” - '
                        'InvestmentMoats<br><br>REACH OUT TO US<br>Please contact '
                        'us if you have any queries via:<br><br>- Our Live Chat '
                        'service on the Syfe app and web dashboard<br>- Email at '
                        'support@syfe.com<br>- Call us at +65 3138 1215 between '
                        '9:00 am and 6:00 pm Monday through Friday<br><br><br>MAS '
                        'Capital Markets Services (CMS) License - CMS100837. '
                        'Located at 4 Robinson Rd, #11-01 The House Of Eden, '
                        'Singapore 048543.',
     'developer': 'Syfe Pte. Ltd.',
     'developerAddress': None,
     'developerEmail': 'support@syfe.com',
     'developerId': 'Syfe+Pte.+Ltd.',
     'developerInternalID': '6414178832405907722',
     'developerWebsite': 'https://www.syfe.com',
     'editorsChoice': False,
     'free': True,
     'genre': 'Finance',
     'genreId': 'FINANCE',
     'headerImage': 'https://play-lh.googleusercontent.com/Yeggbs43FhOOoFdr8FbJs67LjSCQNn45rDg2tNrzexPQIUZqQc7R9rFYrTf5LeWy7A',
     'histogram': [18, 9, 15, 56, 217],
     'icon': 'https://play-lh.googleusercontent.com/wQ8cTcyXnCDZfpu0Ft8vMlqHN5VPtTt_pgkthOYy2okpR4qvu20fhHmme0ldcqPOrmTu',
     'inAppProductPrice': None,
     'installs': '50,000+',
     'minInstalls': 50000,
     'offersIAP': False,
     'originalPrice': None,
     'price': 0,
     'privacyPolicy': 'https://www.syfe.com/privacy-policy',
     'ratings': 315,
     'recentChanges': '1. See statuses for various transactions - withdrawals, '
                      'fund transfers and new funds added\r\n'
                      '2. Download new and past monthly statements from account '
                      'settings\r\n'
                      '3. See more detailed information on portfolios when adding '
                      'new portfolios\r\n'
                      '4. Minor fixes and performance improvements',
     'recentChangesHTML': '1. See statuses for various transactions - withdrawals, '
                          'fund transfers and new funds added<br>2. Download new '
                          'and past monthly statements from account settings<br>3. '
                          'See more detailed information on portfolios when adding '
                          'new portfolios<br>4. Minor fixes and performance '
                          'improvements',
     'released': 'Feb 13, 2020',
     'reviews': 112,
     'sale': False,
     'saleText': None,
     'saleTime': None,
     'score': 4.3960395,
     'screenshots': ['https://play-lh.googleusercontent.com/H58nNrSz9aQldpSObxkOc775H2aUWS8Rwi6etXLF-lL5lpBOvHM5XyADGuVB_4TPr3_5',
                     'https://play-lh.googleusercontent.com/EJuqpmk0aAosjTv81L2EceMf-a7VpQmyd4FVS3-h2kMHLJJMjZD9yUtbUwigho4wOcU',
                     'https://play-lh.googleusercontent.com/2JbHM66KBRzVZr5N8w_i3kn_Eye5Zjps76_BJz2w-0kgr9q2ff6cMxhIv8ICeMy4svg',
                     'https://play-lh.googleusercontent.com/nLWXA0ub8n6FB5lWN9lukn41AyXUr3wh7DEfNRD73q6jIa8ityevjK3B1iUsBvjLKw',
                     'https://play-lh.googleusercontent.com/VjEVGMTcsoMuRUQWSAKfIZbkgstuRXkzdnhkfniUHMPh5lUrniIOzQ_93MlrLPmogoDj',
                     'https://play-lh.googleusercontent.com/p3u0RUMoTqytOqU_XSrWHxnBOkNXuelUXfHogrFZ1yCJ_xbAUOJ3FWwy78uyr0DjacU',
                     'https://play-lh.googleusercontent.com/AB5Gqg8Um5Uj2A7yyflZhAmBgkCz_8PunAOaLpd5k0_FGmkjt8GzINvYJNYjLbmlLNo',
                     'https://play-lh.googleusercontent.com/Zz5ECBOSC2b85h3hS0AlRe2cm1i84HBfwn40QMSEo9PNxlw2c4byGqKbNqqTZrNnwA'],
     'size': '36M',
     'summary': 'Start your investment journey. Invest and grow your wealth with '
                'any amount.',
     'summaryHTML': 'Start your investment journey. Invest and grow your wealth '
                    'with any amount.',
     'title': 'Syfe: Invest Better',
     'updated': 1630425088,
     'url': 'https://play.google.com/store/apps/details?id=com.syfe&hl=en&gl=us',
     'version': '4.80',
     'video': None,
     'videoImage': None}





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>description</th>
      <th>descriptionHTML</th>
      <th>summary</th>
      <th>summaryHTML</th>
      <th>installs</th>
      <th>minInstalls</th>
      <th>score</th>
      <th>ratings</th>
      <th>reviews</th>
      <th>...</th>
      <th>adSupported</th>
      <th>containsAds</th>
      <th>released</th>
      <th>updated</th>
      <th>version</th>
      <th>recentChanges</th>
      <th>recentChangesHTML</th>
      <th>editorsChoice</th>
      <th>appId</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Syfe: Invest Better</td>
      <td>Invest better, faster and smarter with Syfe. S...</td>
      <td>Invest better, faster and smarter with Syfe. S...</td>
      <td>Start your investment journey. Invest and grow...</td>
      <td>Start your investment journey. Invest and grow...</td>
      <td>50,000+</td>
      <td>50000</td>
      <td>4.396039</td>
      <td>315</td>
      <td>112</td>
      <td>...</td>
      <td>None</td>
      <td>False</td>
      <td>Feb 13, 2020</td>
      <td>1630425088</td>
      <td>4.80</td>
      <td>1. See statuses for various transactions - wit...</td>
      <td>1. See statuses for various transactions - wit...</td>
      <td>False</td>
      <td>com.syfe</td>
      <td>https://play.google.com/store/apps/details?id=...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Endowus: Invest CPF, SRS, Cash</td>
      <td>Thousands invest and grow their Cash, CPF &amp; SR...</td>
      <td>Thousands invest and grow their Cash, CPF &amp;amp...</td>
      <td>Endowus offers access to superior investment p...</td>
      <td>Endowus offers access to superior investment p...</td>
      <td>10,000+</td>
      <td>10000</td>
      <td>4.900000</td>
      <td>254</td>
      <td>143</td>
      <td>...</td>
      <td>None</td>
      <td>False</td>
      <td>Oct 6, 2020</td>
      <td>1629383049</td>
      <td>1.3.1</td>
      <td>Upgrade to v1.3.1 to find the latest bug fixes...</td>
      <td>Upgrade to v1.3.1 to find the latest bug fixes...</td>
      <td>False</td>
      <td>com.endowus.mobileapp</td>
      <td>https://play.google.com/store/apps/details?id=...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>StashAway: Invest and save</td>
      <td>StashAway is where intelligent investing meets...</td>
      <td>StashAway is where intelligent investing meets...</td>
      <td>Personal finance and investing</td>
      <td>Personal finance and investing</td>
      <td>100,000+</td>
      <td>100000</td>
      <td>4.346535</td>
      <td>3015</td>
      <td>1464</td>
      <td>...</td>
      <td>None</td>
      <td>False</td>
      <td>Oct 25, 2017</td>
      <td>1629966240</td>
      <td>11.95.2</td>
      <td>A release with no new features?! Why aren’t we...</td>
      <td>A release with no new features?! Why aren’t we...</td>
      <td>False</td>
      <td>com.awp.stashaway</td>
      <td>https://play.google.com/store/apps/details?id=...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 49 columns</p>
</div>



## Scrapping Google Play Store reviews


```python
# for scraping app reviews from Google Play Store
from google_play_scraper import app, Sort, reviews

# Empty list for storing reviews
google_app_reviews = []

## Loop through apps to get reviews
for app_name, app_id in zip(google_app_names, google_app_ids):
    
    # Get start time
    start = dt.datetime.now(tz=get_localzone())
    fmt= "%m/%d/%y - %T %p"    
    
    # Print starting output for app
    print('---'*20)
    print('---'*20)    
    print(f'***** {app_name} started at {start.strftime(fmt)}')
    print()
    
    # Number of reviews to scrape per batch
    count = 200
    
    # To keep track of how many batches have been completed
    batch_num = 0
     
    # Retrieve reviews (and continuation_token) with reviews function
    rvws, token = reviews(
        app_id,           # found in app's url
        lang='en',        # defaults to 'en'
        country='us',     # defaults to 'us'
        sort=Sort.NEWEST, # start with most recent
        count=count       # batch size
    )
    
    # For each review obtained
    for r in rvws:
        r['app_name'] = app_name # add key for app's name
        r['app_id'] = app_id     # add key for app's id
     
    
    # Add the list of review dicts to overall list
    google_app_reviews.extend(rvws)
    
    # Increase batch count by one
    batch_num +=1 
    print(f'Batch {batch_num} completed.')
    
    # Wait 1 to 5 seconds to start next batch
    time.sleep(random.randint(1,5))
    
    # Append review IDs to list prior to starting next batch
    pre_review_ids = []
    for rvw in google_app_reviews:
        pre_review_ids.append(rvw['reviewId'])
    
    # Loop through at most max number of batches
    for batch in range(4999):
        rvws, token = reviews( # store continuation_token
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=count,
            # using token obtained from previous batch
            continuation_token=token
        )
        
        # Append unique review IDs from current batch to new list
        new_review_ids = []
        for r in rvws:
            new_review_ids.append(r['reviewId'])
            
            # And add keys for name and id to each review dict
            r['app_name'] = app_name # add key for app's name
            r['app_id'] = app_id     # add key for app's id
     
        # Add the list of review dicts to main app_reviews list
        google_app_reviews.extend(rvws)
        
        # Increase batch count by one
        batch_num +=1
        
        # Break loop and stop scraping for current app if most recent batch
          # did not add any unique reviews
        all_review_ids = pre_review_ids + new_review_ids
        if len(set(pre_review_ids)) == len(set(all_review_ids)):
            print(f'No reviews left to scrape. Completed {batch_num} batches.\n')
            break
        
        # all_review_ids becomes pre_review_ids to check against 
          # for next batch
        pre_review_ids = all_review_ids
        
        # Wait 1 to 5 seconds to start next batch
        time.sleep(random.randint(1,5))
      
    
    # Print update when max number of batches has been reached
      # OR when last batch didn't add any unique reviews
    print(f'Done scraping {app_name}.')
    print(f'Scraped a total of {len(set(pre_review_ids))} unique reviews.\n')
    
    # Get end time
    end = dt.datetime.now(tz=get_localzone())
    
    # Print ending output for app
    print(f"""
    Successfully inserted all {app_name} reviews into collection
    at {end.strftime(fmt)}.\n
    """)
    print(f'Time elapsed for {app_name}: {end-start}')
    print('---'*20)
    print('---'*20)
    print('\n')
    
    # Wait 1 to 5 seconds to start scraping next app
    time.sleep(random.randint(1,5))
```

    ------------------------------------------------------------
    ------------------------------------------------------------
    ***** Syfe started at 09/04/21 - 23:33:25 PM
    
    Batch 1 completed.
    No reviews left to scrape. Completed 2 batches.
    
    Done scraping Syfe.
    Scraped a total of 110 unique reviews.
    
    
        Successfully inserted all Syfe reviews into collection
        at 09/04/21 - 23:33:31 PM.
    
        
    Time elapsed for Syfe: 0:00:05.276724
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    
    ------------------------------------------------------------
    ------------------------------------------------------------
    ***** Endowus started at 09/04/21 - 23:33:32 PM
    
    Batch 1 completed.
    No reviews left to scrape. Completed 2 batches.
    
    Done scraping Endowus.
    Scraped a total of 250 unique reviews.
    
    
        Successfully inserted all Endowus reviews into collection
        at 09/04/21 - 23:33:34 PM.
    
        
    Time elapsed for Endowus: 0:00:02.298619
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    
    ------------------------------------------------------------
    ------------------------------------------------------------
    ***** StashAway started at 09/04/21 - 23:33:38 PM
    
    Batch 1 completed.
    No reviews left to scrape. Completed 8 batches.
    
    Done scraping StashAway.
    Scraped a total of 1515 unique reviews.
    
    
        Successfully inserted all StashAway reviews into collection
        at 09/04/21 - 23:34:00 PM.
    
        
    Time elapsed for StashAway: 0:00:22.199429
    ------------------------------------------------------------
    ------------------------------------------------------------
    
    



```python
# Converting output to dataframe
google_reviews = pd.DataFrame(google_app_reviews)
```

## Part 3: Combining both Apple and Google Store reviews


```python
print(f'Apple Store: {np.shape(apple_reviews)[0]} rows and {np.shape(apple_reviews)[1]} columns.')
print(f'Google Play Store: {np.shape(google_reviews)[0]} rows and {np.shape(google_reviews)[1]} columns.')
```

    Apple Store: 524 rows and 9 columns.
    Google Play Store: 1515 rows and 12 columns.



```python
new_apple = apple_reviews[['app_name','review']] # Selecting app_name and review from apple reviews into new df
new_apple.rename(columns={'review':'content'},inplace=True) # rename review content column
new_google = google_reviews[['app_name','content']] # subset app_name and content from google reviews into new df
total_reviews = pd.concat([new_apple,new_google]) # Concat both dfs into one

# saving reviews into csv file
total_reviews.to_csv('app_reviews.csv', index=False, header=True)
```

    /Users/a844133yara.com/.pyenv/versions/3.9.5/envs/python_playground/lib/python3.9/site-packages/pandas/core/frame.py:5034: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().rename(

