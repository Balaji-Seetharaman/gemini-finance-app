# Gemini Stock Analyzer

Gemini Stock Analyzer is a Streamlit-based web application that provides comprehensive stock analysis using Google's Gemini AI model. The app offers various features to help users analyze stocks, including sentiment analysis, financial metrics, and the ability to chat with stock data and annual reports.

## Features

1. **Sentiment Analysis**: Analyze the market sentiment of a stock based on recent news articles.
2. **Stock Score**: Calculate a stock score based on key financial metrics like P/E ratio, P/B ratio, dividend yield, and ROE.
3. **Stock Report**: Generate a detailed stock analysis report that can be downloaded as a PDF.
4. **Chat with 10K Form**: Ask questions about a company's annual report by uploading a 10K PDF file.
5. **Chat with Stock Data**: Interact with historical stock data through natural language queries.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/gemini-stock-analyzer.git
   cd gemini-stock-analyzer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Navigate through the different features using the sidebar:
- Company Analysis: Enter a stock symbol to get sentiment analysis, stock score, and a detailed report.
- Ask Question on Annual Report: Upload a 10K PDF file and ask questions about it.
- Chat with Data: Enter a stock symbol and ask questions about its historical data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for educational purposes only. Always conduct your own research and consult with a financial advisor before making investment decisions.
