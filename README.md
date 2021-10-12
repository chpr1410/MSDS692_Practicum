# Overview
This is the first practicum project for my MS in Data Science.  This project investigates whether fundamentally healthy companies experience better stock market returns than market indexes in general.  Further, I experiment with enhancing this strategy with machine learning methods.
# Motivation
The stock market is a huge part of our economy.  Making money in the market can be a risky proposition and traditionally has been viewed as a realm in which only the Wall Street elites have experienced success.  But in the last few years, many online brokerages have enhanced their technologies and transitioned to zero-commission models, making the stock market more accessible than ever.  

However, accessibility does not equal success.  Now that virtually anyone can trade or invest in stocks, finding profitable strategies – or at least ones that mitigate risk while still allowing a person to build their wealth – is the next major obstacle for self-directed investors.   

I have been interested in the stock market for several years.  Prior coursework in accounting and finance fueled this interest, but it did so in a different way than I saw from most of my peers.  While lots of 21st century investors are enamored with technical trading – using historical prices, technical indicators, and quickly-correcting arbitrage to try and predict price changes - I found myself more and more interested in fundamental investing.  

Fundamental investing focuses on, well, the fundamentals.  These include how well the company has been performing and growing in the recent past, according to the financial statements, or how much cash and short-term assets it can use to deal with future uncertainties, to name a couple.

In short, fundamental investing focuses on the intrinsic value of the underlying business and the financial health of the company.  

Once I grasped the difference of philosophies, I realized that some companies are set up to perform better than others, and that there are ways to quantitatively determine which companies these are. Two books were fundamental in shaping that perspective.  One is the seminal value investing book "The Intelligent Investor" by Benjamin Graham, and the other is an updated and simplfied version of the former, called "Rule #1 Investing", by Phil Town. They both introduced me to the theory behind using fundamentals to invest. 

Naturally, I wanted to test that theory in an empirical manner.  I wondered how well the stocks of these financially healthy companies performed over time in comparison to the broader market. 

This project was the perfect opportunity to find that answer.  

# Description of the Fundamental Data

The fundamental data for this project was provided by [QuickFS](https://quickfs.net/).

QuickFS, short for Quick Financial Statements, provides historical balance sheet, income statement, cash flow statement, and financial ratios and other metrics for almost 33,000 companies.  Where applicable, the historic data dates back 20 years into the past.  

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/qfs%20supported%20cos.JPG)

QuickFS provides quarterly and annual data for each company.  QuickFS has around 280 metrics available to download per company per period.  I focused on the following 9 metrics:

- Original Filing Date
- Return on Invested Capital
- Equity Growth
- Earnings Per Share Growth
- Revenue
- Revenue Growth
- Free Cash Flow
- Free Cash Flow Growth
- Long-Term Debt

I feature engineered some metadata and additional growth rates from the above metrics.  In total, I stored around 20 features for each company.

The total number of observations - the aforementioned metrics for each company for 20 years and for both quarterly and annual periods - amounted to around 23 million data points.

# Filtering Process
Once I had this fundamental data stored in a database, I put each filing date for each company through the following filter:

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/healthy%20co%20filtering%20process.JPG)

Each filing period that makes it through the filter is classified as a "Healthy Company", and stored in a list.  After removing duplicates arising from the quarterly and annual datasets, 619 instances of healthy companies remained.

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/healthy%20cos%20by%20year.JPG)

# Pricing Process

At this stage of the project, I had a list of companies and at what date they were considered healthy.  At this milestone, I could download stock price data for these companies on these dates - as well as for dates in the future - and see exactly how these healthy companies' stock prices performed.

I downloaded pricing data from three sources: 
- [Alpha Vantage](https://www.alphavantage.co/)
- [Financial Modeling Prep](https://financialmodelingprep.com/)
- [Yahoo Finance](https://finance.yahoo.com/)

For each company and date pair, I downloaded the stock price for the healthy filing date and then future periods of a week, month, six months, 1 year, 1.5 years, and 2 years. As some of these periods would be beyond today's date, the pricing data was not available for each period for each instance of healthy company.    

# How does the strategy Perform?

Once I had pricing data, I could finally determine how well these healthy companies performed over time.  The results are encouraging!

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/healthy%20co%20returns%20over%20time.JPG)

Each period is positive and an average one year return of 22.89% for each stock is a very solid return on investment.

Separating the companies by year allows me to see what performance would have been like over time.  From 2009 to the current period, there are a couple highly positive years, one negative year, and several good years sprinkled in. 

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/healthy%20co%20returns%20by%20year.JPG)


### Comparison to Major Indices

So, how does this strategy of investing in healthy companies compare to the performance of the major indices over time?

The answer is: **quite favorably**.  The table below shows that the strategy has an average annual return that is almost double the major indices.  Compounding this over time makes a huge difference.  A dollar invested in this healthy company strategy in 2009 results in a current ending balance 3 to 4 times as large as the same dollar invested in the major indices. 

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/comparing%20strategy%20to%20indices.JPG)


# Using Machine Learning to Improve the Strategy

The results thus far have been extremely positive.  But as a blossoming data scientist, isn't it my responsibility to ask: **Can machine learning improve the strategy even further???**

To find out, I take the list of healthy companies, download additional financial metrics from QuickFS, and then put this data into several different ML models to see if they can improve the strategy.

### Problem Definition

I define this machine learning problem as a binary classification one.  Essentially, I separate the results of the healthy companies into two categories, 0s and 1s where:

- **0** = a healthy company that provided a one year return of less than or equal to 10%
- **1** = a healthy company that provided a one year return of greater than 10%
 

### The Data

For each healthy company and filing period pair, I downloaded 10 years of 29 additional financial metrics.  These metrics include balance sheet ratios like debt-to-equity and pricing ratios like price-to-earnings.  The full list is available in the main code in this repository.

After downloading this additional data, I went through a cleaning process, checked for class balance, and scaled each feature.  Finally, I split it into training and test sets, and I was ready to try out different models.

### The Models

I experimented with 5 models:

**1.) TPOT Automation**
The TPOT package acts as a data science assistant, trying several different ML algorithms and hyperparameters.  I ran the TPOT pipeline for a few hours and at the end the optimal model was stored.

**2.) AutoKeras**
Similar to TPOT, AutoKeras automates model selection.  AutoKeras, however, works with artificial neural networks.  I ran the AutoKeras optimizer and saved the best model.

**3.) Multilayer Perceptron (MLP)**

I built a custom sequential Keras model utilizing dense layers, L1 regulation, batch normalization and dropout.

**4.) Random Forest (RF)**

A simple implementation of the popular algorithm using SciKit Learn.  

**5.) Convolutional Neural Network (CNN)**

Lastly, I created a 2D CNN to analyze the data.  I restructured the input data into 2D matrices, similar to what CNNs are normally used for in terms of image classification.  

## Model Performance Comparison

I trained the models and then test them on the test set.  I evaluated each model on accuracy, Area Under the ROC curve, and from the computation of what percent return the strategy would make on the test set compared to what a passive strategy would make. 

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/model%20evaluation%20scores.JPG)

The MLP and CNN models have the best evaluations.  They have the highest accuracies and ROC scores, as well as the best strategy returns.  The CNN reached the highest accuracy of 72%.  All the models seem to predict fairly well, especially compared to a random baseline model which in theory, would get close to a 50% accuracy.  Each model also improves on the baseline.  For the test set, the strategy returns are over 20% while the passive, baseline strategy is around 14%.

I further evaluate the models by using confusion matrices.  Here is the summary:

![Test Image 4](https://github.com/chpr1410/MSDS692_Practicum/blob/main/app/static/images/confusion%20matrices.JPG)

These matrices correspond to the close performance of the models above.  The CNN identified the most postive companies, while the MLP identified the most negative. I was curious to look at false positives, though.  Investing in a company that the model says is positive, but turns out to be negative would pose a significant detriment to an investing strategy.  Avoiding these false positives would be a main priority.  The MLP model does this well.

I was pleased to see that following each model's strategy resulted in a higher return than just passively investing in all healthy companies.  Keeping in mind that this is only for a limited sample - the test set - it is still very encouraging that machine learning enhances the strategy of investing in healthy companies.

# Conclusions

For this project I have introduced sources of fundamental and pricing data for stocks, and how to query the APIs to obtain and store the data.

After defining criteria for a healthy company, I found 619 instances of healthy companies since 2009.  Of those instances, 479 qualified at least a year ago, and therefore have a one year return percentage.  The average one year return percentage for these companies is 22.89%.

Then, through separating these healthy companies into their respective years, I have shown that investing in healthy companies over the period of 2009 to the present results in a weighted average one year return of 24.35%.  This return is significantly higher than each of the market indices.  

Finally, I have also shown how to implement machine learning to augment the base strategy.  I found that all the machine learning methods that I applied increased average one year returns by 8.5%.

# Further Research Opportunities

Using a larger dataset could further demonstrate the value of this approach.  The dataset I used had historical data going back 20 years.  The strategy requires 10 years of solid financial performance, so that leaves only 10 years of potential investments.  A dataset encapsulating more years would provide more opportunities to evaluate this strategy, and in different economic climates.

The dataset was also limited in its geography.  The United States was the main focus, which has merit, but I would be interested to see how this strategy performs across Asian, European, and South American markets.  As globalization continues, investing opportunities can arise anywhere in the world. 

Also, more sophisticated machine learning methods could be applied.  The models I used were fairly simple and could no doubt be improved upon.  And in reference to the previous points, combining more complex models with a more robust dataset could provide even better evaluations than I achieve here.  

Other data types were outside the scope of this project, but further research could be conducted on combining this fundamental analysis with methods like the natural language processing of news articles or annual reports to glean further insight into company health.

*** Disclaimer: I am not a financial advisor and the findings presented here are for educational purposes only and should not be taken as investment advice.
