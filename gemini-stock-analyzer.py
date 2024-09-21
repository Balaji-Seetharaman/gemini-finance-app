import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import base64


import yfinance as yf 
import pandas as pd
from GoogleNews import GoogleNews
import io
import sys
import matplotlib.pyplot as plt
import numpy as np
from streamlit_echarts import st_echarts
from io import BytesIO
from fpdf import FPDF
from PyPDF2 import PdfReader




load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])




st.set_page_config(page_title="Gemini Stock Analyzer")


sentiment_mapping = {
    "Bullish": 1,
    "Bearish": -1,
    "Neutral": 0
}



def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    pe_ratio = info.get('trailingPE')
    pb_ratio = info.get('priceToBook')
    dividend_yield = info.get('dividendYield') * 100 if info.get('dividendYield') else 0
    roe = info.get('returnOnEquity') * 100 if info.get('returnOnEquity') else 0

    return pe_ratio, pb_ratio, dividend_yield, roe

def calculate_score(pe_ratio, pb_ratio, dividend_yield, roe):
    score = 0
    pe_weight = 0.25
    pb_weight = 0.25
    dividend_weight = 0.25
    roe_weight = 0.25

    pe_score = (1 / pe_ratio) * pe_weight if pe_ratio and pe_ratio > 0 else 0
    pb_score = (1 / pb_ratio) * pb_weight if pb_ratio and pb_ratio > 0 else 0
    dividend_score = dividend_yield * dividend_weight
    roe_score = roe * roe_weight
  
    score = pe_score + pb_score + dividend_score + roe_score

    return score
    

def draw_semi_circular_gauge(value):
    display_value = value * 100

    option = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "radius": "100%",
                "center": ["50%", "75%"],
                "axisLine": {
                    "lineStyle": {
                        "width": 30,
                        "color": [
                            [0.5, "#ff0000"],  
                            [1, "#00ff00"],   
                        ],
                    },
                },
                "pointer": {
                    "length": "70%",
                    "width": 8,
                },
                "min": -100,
                "max": 100,
                "splitNumber": 10,
                "axisLabel": {
                    "show": False
                },
                "axisTick": {
                    "show": False
                },
                "splitLine": {
                    "show": False
                },
                "detail": {
                    "formatter": "{value}%",
                    "offsetCenter": [0, "30%"],
                    "fontSize": 20,
                },
                "data": [{"value": display_value}],
            }
        ],
        "title": {
            "show": True,
            "offsetCenter": [0, "100%"]
        },
    }
    st_echarts(options=option)


def is_valid_us_stock(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        if stock_info.get('exchange') in ['NMS','NYQ']:
            return True
        else:
            return False
    except Exception as e:
        return False

def extract_google_news(stock_ticker, num_articles=5):
    googlenews = GoogleNews(lang='en')
    googlenews.search(f"{stock_ticker} stock")
    
    googlenews.getpage(1)
    
    news_results = googlenews.results(sort=True)
    
    news = []
    for entry in news_results[:num_articles]:
        title = entry['title']
        link = entry['link']
        news_title = f"[{title}]({link})"
        news.append({'news_title': news_title})
    
    return news


def generate_sentiment_prompt(title):
    prompt = f"Analyse the market sentiment of the news title - {title} and return only one of the following sentiment Bullish, Bearish or Netural and don't provide any explanations"
    return prompt

def generate_pdf(stock_analysis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    pdf.multi_cell(0, 10, txt=stock_analysis)
    pdf_output_path = "stock_analysis.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path


def download_max_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="max")
    
    data.index = data.index.strftime('%m/%d/%Y')
    
    filename = f"{symbol}_max_data.csv"
    data.to_csv(filename)
    st.write(f"Data for the stock {symbol} is  downloaded")




def pdf_qa(pdf_file, prompt):
    buffer = io.StringIO()
    sys.stdout = buffer
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    pdf_reader = PdfReader(pdf_file)
    document_text = ""
    
    for page in pdf_reader.pages:
        document_text += page.extract_text() + "\n"

    qa_prompt = f"""You are a helpful assistant specialized in financial documents. 
    The user has uploaded a 10K document, and they have the following question: 
    '{prompt}' 
    Please provide a detailed and concise answer based on the contents of the document
    document: {document_text}"""

    responses = model.generate_content( qa_prompt)

    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()

def predict_sentiment(prompt):
    buffer = io.StringIO()
    sys.stdout = buffer
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    responses = model.generate_content(prompt)
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()


def qa_agent_df(user_prompt,symbol):
    buffer = io.StringIO()
    sys.stdout = buffer

    filename = f"{symbol}_max_data.csv"
    if not os.path.isfile(filename):
        download_max_data(symbol)

    df = pd.read_csv(f'{symbol}_max_data.csv')
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    

    prompt = f"""
        The following is a table of data:
        
        {df}
        
        Columns: {', '.join(df.columns)}
        
        Please answer the question based on the table data.
        
        Question: {user_prompt}
        Answer:
    """

    responses = model.generate_content(prompt)
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()





def generate_stock_analysis(symbol):
    buffer = io.StringIO()
    sys.stdout = buffer

    prompt= f"""
        Perform a comprehensive analysis of {symbol} Include:
        Overview: Primary business operations and market position.
        Financials: Revenue, profit margins, EPS, growth projections, and balance sheet.
        Valuation: P/E ratio, P/B ratio, and other relevant metrics.
        Market Performance: Stock price trends and market sentiment.
        Risks: Operational, market, and financial risks.
        Competition: Industry comparison and competitive strengths/weaknesses.
        ESG Factors: ESG ratings and sustainability initiatives.
        Outlook: Recent developments and future plans.
        Use the latest Market Analysis data and insights. Present information clearly.
    """

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    responses = model.generate_content(prompt)
    try:
        for response in responses:
            print(response.text, end="")
    finally:
        sys.stdout = sys.__stdout__
    
    return buffer.getvalue()




def main_page():
    st.sidebar.title('Gemini Finance App')

    mapping = {'company_analysis': '📈 Dashboard',
            'chat_on_pdf': '❓ Ask Question on Annual Report',
            'chat_with_data': ' 📊 Chat with Data'
            }

    col = st.columns((6, 6), gap='large')

    selected_tab = None
    selected_tab = st.sidebar.radio(label='Go to', options=("company_analysis", "chat_on_pdf","chat_with_data"), format_func=lambda x: mapping[x],
                                        label_visibility='hidden')

    if selected_tab == 'company_analysis':
        st.subheader('Company Data')
        symbol_input = st.text_input("Enter a stock symbol:")
        
        if st.button("Submit",key="stock_button"):

            if is_valid_us_stock(symbol_input):
                valid_stock = 1
                st.write(f"Please wait while we fetch the analysis of stock {symbol_input}")
            else:
                st.write(f"{symbol_input} is not a valid US stock symbol. Please enter a valid symbol.")
                return 

            tab1, tab2,tab3 = st.tabs(['Sentiment Analysis', 'Stock Score', 'Stock Report'])
            with tab1:
                if valid_stock: 

                    news = extract_google_news(symbol_input, num_articles=5)
                    df = pd.DataFrame(news)

                    df['sentiment'] = df['news_title'].apply(lambda title: predict_sentiment(generate_sentiment_prompt(title)))
                    df['sentiment'] = df['sentiment'].str.replace('\n', '').str.replace(' ', '')

                    df['sentiment_value'] = df['sentiment'].map(sentiment_mapping).fillna(0).astype(int)
                    print(df)
                    sentiment_score = df['sentiment_value'].mean()
                    value = 'Bullish' if sentiment_score > 0 else ('Bearish' if sentiment_score < 0 else 'Neutral')
                    st.info(f'Sentiment for the Stock {symbol_input} is {value}')

                    st.write(f'{value} Sentiment  for the Stock {symbol_input} is {int(sentiment_score * 100)} %')
    
                    
                    draw_semi_circular_gauge(sentiment_score)

                    st.title('Bullish/Bearish Sentiment Meter')

            with tab2:
                try:
                    pe_ratio, pb_ratio, dividend_yield, roe = fetch_financial_data(symbol_input)

                    print(f"Financial Data for {symbol_input}:")
                    print(f"  P/E Ratio: {pe_ratio}")
                    print(f"  P/B Ratio: {pb_ratio}")
                    print(f"  Dividend Yield: {dividend_yield}%")
                    print(f"  ROE: {roe}%")

                    score = calculate_score(pe_ratio, pb_ratio, dividend_yield, roe)
                    st.info(f'Stock Score: {score:.2f}')

                    if score > 1.0:
                        st.info('Recommendation: Buy')
                        print("Recommendation: Buy")
                    elif 0.5 <= score <= 1.0:
                        st.info('Recommendation: Hold')
                    else:
                        st.info('Recommendation: Sell')
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")

            with tab3:
                stock_analysis = generate_stock_analysis(symbol_input)
                print(stock_analysis)

              
                pdf_path = generate_pdf(stock_analysis)
                st.success("Generated stock_analysis.pdf")
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, "stock_analysis.pdf")

    
    if selected_tab == 'chat_on_pdf':
        st.title('Chat with 10K Form')
        st.header("Drop the PDF File here")

        pdf_file = st.file_uploader("Drag and drop your PDF file here", type="pdf")
    
        if pdf_file is not None:
            st.success("Thank you for uploading the PDF file.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        try:
            if prompt := st.chat_input("Ask your questions related to attached 10K document for the particular stock"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = pdf_qa(pdf_file,prompt)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
    
    
    if selected_tab == 'chat_with_data':
        st.title('Chat with data')
        with st.sidebar:
            symbol_input = st.text_input("Enter a stock symbol data that you want to chat on:")

            if st.button("Submit"):
                try:
                    if is_valid_us_stock(symbol_input):
                        download_max_data(symbol_input)    
                    else:
                        st.write(f"{symbol_input} is not a valid US stock symbol. Please enter a valid symbol.")
                except Exception as e:
                    st.error(f"An error occurred while processing the stock data: {e}")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        try:
            if user_prompt := st.chat_input("Ask your questions related to the stock data you submitted"):
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    response = qa_agent_df(user_prompt, symbol_input)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")
        



def main():
    main_page()




if __name__ == "__main__":
    main()
