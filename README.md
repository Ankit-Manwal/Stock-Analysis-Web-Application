# 📈 Stock Analysis Web Application

A comprehensive web application for stock market analysis, prediction, and portfolio management built with Flask and machine learning.

## 🌟 Features

### 📊 Stock Analysis
- **Real-time Stock Data**: Get live stock prices and historical data using Yahoo Finance API
- **Interactive Charts**: Multiple chart types including candlestick, line graphs, and volume analysis
- **Technical Indicators**: EMA (Exponential Moving Average) analysis with 20-50 and 100-200 periods
- **Stock Comparison**: Compare multiple stocks side by side

### 🤖 AI-Powered Predictions
- **Deep Learning Model**: LSTM-based neural network for stock price prediction
- **Future Price Forecasting**: Predict stock prices for the next 30 days
- **Trend Analysis**: Visualize predicted vs actual price trends
- **Model Performance**: Built-in model evaluation and accuracy metrics

### 📰 Market Intelligence
- **Latest News**: Real-time stock-related news from multiple sources
- **Market Sentiment**: News analysis and sentiment tracking
- **Top Gainers/Losers**: Daily market movers and shakers
- **Market Overview**: Comprehensive market statistics

### 👤 User Management
- **User Authentication**: Secure login and registration system
- **Personal Watchlist**: Add/remove stocks from your personal watchlist
- **User Reviews**: Rate and review stocks with community feedback
- **Session Management**: Secure session handling with Flask-Session

### 🎨 Modern UI/UX
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Bootstrap Integration**: Modern, clean interface with Bootstrap 5
- **Interactive Elements**: Smooth animations and user-friendly navigation
- **Dark/Light Theme**: Customizable appearance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- MongoDB (running locally on port 27017)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Stock-Analysis-Web-Application.git
   cd Stock-Analysis-Web-Application
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   # On Windows
   env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up MongoDB**
   - Install MongoDB locally
   - Start MongoDB service
   - The application will automatically create the required database and collections

5. **Run the application**
   ```bash
   python main_app.py
   ```

6. **Access the application**
   - Open your browser and go to `http://localhost:5000`

## 📁 Project Structure

```
Stock-Analysis-Web-Application/
├── main_app.py                 # Main Flask application
├── requirements.txt            # Python dependencies
├── stock_dl_model.h5          # Pre-trained LSTM model
├── .gitignore                 # Git ignore file
├── README.md                  # This file
├── templates/                 # HTML templates
│   ├── home.html             # Landing page
│   ├── stock_show.html       # Stock analysis page
│   ├── prediction.html       # Prediction results
│   ├── news.html            # News page
│   ├── watchlist.html       # User watchlist
│   ├── login.html           # Login page
│   ├── signUp.html          # Registration page
│   └── ...
├── static/                   # Static assets
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   └── images/              # Images and charts
└── flask_session/           # Session storage
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
SECRET_KEY=your_secret_key_here
MONGODB_URL=mongodb://127.0.0.1:27017/
```

### MongoDB Setup
The application uses MongoDB for user data and reviews. Make sure MongoDB is running on `localhost:27017`.

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/explore` | GET/POST | Market exploration |
| `/stock_show/<symbol>` | GET | Stock analysis page |
| `/stock_show/predict/<symbol>` | GET | Stock prediction |
| `/stock_show/<symbol>/news` | GET | Stock news |
| `/stock_show/<symbol>/reviews` | GET/POST | Stock reviews |
| `/watchlist` | GET | User watchlist |
| `/add_watchlist` | POST | Add stock to watchlist |
| `/remove_watchlist` | POST | Remove stock from watchlist |
| `/login` | GET/POST | User login |
| `/signUp` | GET/POST | User registration |
| `/logout` | GET | User logout |

## 🤖 Machine Learning Model

### Model Architecture
- **Type**: LSTM (Long Short-Term Memory) Neural Network
- **Framework**: Keras/TensorFlow
- **Input**: Historical stock price data (OHLCV)
- **Output**: Future price predictions (30 days)
- **Preprocessing**: MinMaxScaler for data normalization

### Model Training
The model is pre-trained on historical stock data and saved as `stock_dl_model.h5`. To retrain:
1. Collect historical data for your target stocks
2. Preprocess the data using the same pipeline
3. Train the LSTM model
4. Save the model as `stock_dl_model.h5`

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **Flask-Login**: User authentication
- **Flask-Session**: Session management
- **MongoDB**: Database
- **PyMongo**: MongoDB driver

### Data Science & ML
- **TensorFlow/Keras**: Deep learning
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### APIs & Data Sources
- **yfinance**: Yahoo Finance API
- **nselib**: NSE (National Stock Exchange) data
- **nsetools**: NSE tools and utilities

### Frontend
- **Bootstrap 5**: CSS framework
- **Plotly**: Interactive charts
- **Font Awesome**: Icons
- **Google Fonts**: Typography

## 📈 Usage Examples

### Stock Analysis
1. Search for a stock symbol (e.g., "AAPL", "MSFT")
2. View detailed analysis with multiple chart types
3. Analyze technical indicators (EMA, volume)
4. Download data as CSV

### Stock Prediction
1. Navigate to the prediction page for any stock
2. View the AI-generated price forecast
3. Compare predicted vs actual trends
4. Analyze prediction confidence

### Watchlist Management
1. Create an account and log in
2. Add stocks to your personal watchlist
3. Track multiple stocks in one place
4. Remove stocks when no longer interested

## 🔒 Security Features

- **Password Hashing**: bcrypt for secure password storage
- **Session Security**: Signed session cookies
- **Input Validation**: Server-side validation for all inputs
- **CSRF Protection**: Built-in Flask CSRF protection

## 🚀 Deployment

### Local Development
```bash
python main_app.py
```

### Production Deployment
1. Set up a production WSGI server (e.g., Gunicorn)
2. Configure a reverse proxy (e.g., Nginx)
3. Set up MongoDB in production
4. Configure environment variables
5. Set up SSL certificates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/yourusername/Stock-Analysis-Web-Application/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data API
- NSE for Indian market data
- The open-source community for the amazing libraries used
- Contributors and users who provide feedback

## 📊 Screenshots

*Add screenshots of your application here*

---

**Note**: This application is for educational and research purposes. Always do your own research before making investment decisions. Past performance does not guarantee future results.