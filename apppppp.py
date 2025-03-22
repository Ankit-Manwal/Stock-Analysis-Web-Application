import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from nselib import capital_market

import bcrypt
from datetime import datetime, timezone
from bson import ObjectId
import re
import base64

from flask import Flask, render_template, request, send_file, jsonify, session, redirect, url_for, g, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

import arrow
now_india = arrow.now('Asia/Kolkata')


from pymongo import MongoClient 


plt.style.use("fivethirtyeight")

app = Flask(__name__)


from flask_session import Session
# Secret key for signing the session cookie
app.config['SECRET_KEY'] = 'your_secret_key'
# Flask-Session configuration
app.config['SESSION_TYPE'] = 'filesystem'  # Could also be 'redis', 'mongodb', etc.
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
# Initialize Flask-Session
Session(app)




# Load the model (make sure your model is in the correct path)
current_directory = os.getcwd()
model_path = os.path.join(current_directory, 'stock_dl_model.h5')
model = load_model(model_path)

#*****************************************************************************************************************

mongourl= "mongodb://127.0.0.1:27017/"
# Set up MongoDB connection 
client = MongoClient(mongourl) 
db = client['stock'] 
users_collection = db['users'] 



#*****************************************************************************************************************

@app.before_request
def before_request():
    # Store global variable in g (for example, user info)
    g.current_user = current_user  # Store logged-in user or None

# @app.before_request
# def method_override():
#     if request.method == 'POST' and '_method' in request.form:
#         method = request.form['_method'].upper()
#         if method in ['PUT', 'DELETE']:
#             request.environ['REQUEST_METHOD'] = method

#*****************************************************************************************************************


def get_top_gainers_and_losers(date):
    try:
        # Fetch data for the given date
        all_stocks_data = capital_market.bhav_copy_with_delivery(date)
        
        df = pd.DataFrame(all_stocks_data)

        df['PERCENT_CHANGE'] = ((df['CLOSE_PRICE'] - df['PREV_CLOSE']) / df['PREV_CLOSE']) * 100

        top_gainers = df.nlargest(15, 'PERCENT_CHANGE')

        top_losers = df.nsmallest(15, 'PERCENT_CHANGE')

        return {
            "top_gainers": top_gainers[['SYMBOL', 'PERCENT_CHANGE', 'PREV_CLOSE', 'CLOSE_PRICE']],
            "top_losers": top_losers[['SYMBOL', 'PERCENT_CHANGE', 'PREV_CLOSE', 'CLOSE_PRICE']]
        }

    except Exception as e:
        return {"error": str(e)}




@app.route('/explore', methods=['GET', 'POST'])
def gainer_loser():
    date = '3-1-2025'
    if request.method == 'POST':
        date = request.form['date']

    result = get_top_gainers_and_losers(date)

    if "error" not in result:
        print("\n\n**************************************************************************\n\n",result.get("top_gainers", []))
        print("\n\n**************************************************************************\n\n",result.get("top_losers", []))
        print("\n\n**************************************************************************\n\n")

        return render_template('gainer_loser.html', 
                        top_gainers=result.get("top_gainers", []).to_dict(orient='records'),
                        top_losers=result.get("top_losers", []).to_dict(orient='records')
                        )
    
    else:
        return f"Error: {result['error']}"





@app.route('/stock_show', methods=['GET', 'POST'])
@app.route('/stock_show/<string:stock_symbol>', methods=['GET'])
def stock_show(stock_symbol=None):
    # Determine the stock symbol from POST or URL
    if request.method == 'POST':
        stock = request.form.get('stock', 'POWERGRID.NS')  # Default if no input
    elif stock_symbol:
        stock = stock_symbol
    else:
        stock = 'POWERGRID.NS'  # Default stock if no input or URL parameter
    
    # Define the start and end dates for stock data
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2024, 10, 1)
    
    # Download stock data
    df = yf.download(stock, start=start, end=end)
    
    # Descriptive Data
    data_desc = df.describe()
    
    # Exponential Moving Averages
    ema20 = df.Close.ewm(span=20, adjust=False).mean()
    ema50 = df.Close.ewm(span=50, adjust=False).mean()
    ema100 = df.Close.ewm(span=100, adjust=False).mean()
    ema200 = df.Close.ewm(span=200, adjust=False).mean()
    
    # Data splitting
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
    
    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    
    # Prepare data for prediction
    past_100_days = data_training.tail(100)

    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = scaler.fit_transform(final_df)
    
    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions
    y_predicted = model.predict(x_test)
    
    # Inverse scaling for predictions
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    
    # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.Close, 'y', label='Closing Price')
    ax1.plot(ema20, 'g', label='EMA 20')
    ax1.plot(ema50, 'r', label='EMA 50')
    ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.legend()
    ema_chart_path = f"static/{stock}_ema_20_50.png"
    fig1.savefig(ema_chart_path)
    plt.close(fig1)
    
    # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df.Close, 'y', label='Closing Price')
    ax2.plot(ema100, 'g', label='EMA 100')
    ax2.plot(ema200, 'r', label='EMA 200')
    ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.legend()
    ema_chart_path_100_200 = f"static/{stock}_ema_100_200.png"
    fig2.savefig(ema_chart_path_100_200)
    plt.close(fig2)
    
    # Plot 3: Prediction vs Original Trend
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(y_test, 'g', label="Original Price", linewidth = 1)
    ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth = 1)
    ax3.set_title("Prediction vs Original Trend")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Price")
    ax3.legend()
    prediction_chart_path = f"static/{stock}_prediction.png"
    fig3.savefig(prediction_chart_path)
    plt.close(fig3)
    
    # Save dataset as CSV
    csv_file_path = f"static/{stock}_dataset.csv"
    df.to_csv(csv_file_path)

    # # Return the rendered template with charts and dataset
    return render_template('stock_show.html', 
                            stockname=stock,
                            plot_path_ema_20_50=f'{stock}_ema_20_50.png', 
                            plot_path_ema_100_200=f'{stock}_ema_100_200.png', 
                            plot_path_prediction=f'{stock}_prediction.png', 
                            data_desc=data_desc.to_html(classes='table table-bordered'),
                            dataset_link=f'{stock}_dataset.csv'
                          )
        # return render_template(
        # 'zz.html',
        # plot_path_ema_20_50=ema_chart_path,
        # plot_path_ema_100_200=ema_chart_path_100_200,
        # plot_path_prediction=prediction_chart_path,
        # data_desc=data_desc.to_html(classes='table table-bordered'),
        # dataset_link=csv_file_path.split('/')[-1]  # Pass only the filename
        #)

    # return render_template('stock_show.html')


@app.route('/download/<filename>')
def download_file(filename):
    # Ensure that the filename is safe and points to the correct file in the static folder
    filepath = os.path.join('static', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return "File not found", 404




#*****************************************************************************************************************

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/login'  # Redirect here if not logged in

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username=None, watchlist=None):
        self.id = str(user_id)  # Flask-Login requires `id` to be a string
        self.username = username
        self.watchlist=watchlist

# User Loader Function for Flask-Login
@login_manager.user_loader 
def load_user(user_id):
    try:
        # Convert user_id back to ObjectId to query MongoDB
        user_data = users_collection.find_one({'_id': ObjectId(user_id)})
        if user_data:
            #returned to current_user =>  current_user.username ,current_user.id  globally
            return User(user_id=user_data['_id'], username=user_data['username'], watchlist=user_data.get('watchlist', []))
    except Exception as e:
        print(f"Error loading user: {e}")
    return None

#*****************************************************************************************************************


# Utility function to hash a password
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed= bcrypt.hashpw(password.encode('utf-8'), salt)
    # Base64 encode the bcrypt hash for storage
    base64_hashed_password = base64.b64encode(hashed).decode('utf-8')
    return  base64_hashed_password



# Utility function to check a password
# Function to check password (mock)
def check_password(stored_hash, password):
    # Decode the Base64 encoded hash
    decoded_hash = base64.b64decode(stored_hash + '==')  # Ensure correct padding for Base64
    return bcrypt.checkpw(password.encode('utf-8'), decoded_hash)


# Utility function to convert MongoDB ObjectId to string
def str_to_objectid(str_id):
    try:
        return ObjectId(str_id)
    except Exception:
        return None


def validate_user_data(data):
    required_fields = ['username', 'email', 'password']
    errors = []

    # Check for missing required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate email format
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if 'email' in data and not re.match(email_regex, data['email']):
        errors.append("Invalid email format!")

    # Validate username length
    if 'username' in data and len(data['username']) < 3:
        errors.append("Username must be at least 3 characters long!")

    # Validate password length
    if 'password' in data and len(data['password']) < 6:
        errors.append("Password must be at least 6 characters long!")

    # Validate password confirmation
    if 'password' in data and 'confirmPassword' in data and data['password'] != data['confirmPassword']:
        errors.append("Passwords do not match!")

    return len(errors) == 0, errors





@app.route('/signUp', methods=['GET', 'POST'])
def add_user():
    if request.method == 'GET':
        return render_template('signUp.html')
    
    else:

        data = {
            "username": request.form.get("username"),
            "email": request.form.get("email"),
            "password": request.form.get("password"),
            "confirmPassword": request.form.get("confirmPassword"),
            "age": request.form.get("age"),
            "street": request.form.get("street"),
            "city": request.form.get("city"),
            "state": request.form.get("state"),
            "country": request.form.get("country"),
        }

        # Validate user data
        is_valid, errors = validate_user_data(data)
        if not is_valid:
            return jsonify({"errors": errors}), 400

        # Hash the password
        hashed_password = hash_password(data['password'])
        # Create the user object
        user = {
            "username": data['username'],
            "email": data['email'],
            "hashed_password": hashed_password,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        # Optional fields (address-related)
        if 'street' in data and data['street'].strip():
            user['street'] = data['street']
        if 'city' in data and data['city'].strip():
            user['city'] = data['city']
        if 'state' in data and data['state'].strip():
            user['state'] = data['state']
        if 'country' in data and data['country'].strip():
            user['country'] = data['country']

        # Insert user into the database
        try:
            users_collection.insert_one(user)
        except Exception as e:
            return jsonify({"error": "Failed to insert user into the database", "details": str(e)}), 500
        
        print( jsonify({"message": "User created successfully!"}), 201)
        return redirect(url_for('gainer_loser'))


@app.route('/login', methods=['GET' ,'POST'])
def login():
    if request.method == 'GET':
        return render_template('/login.html')
    
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')

    print("\n\n****************************************\n\n",username)
    print("\n\n****************************************\n\n",password)
    print("\n\n****************************************\n\n",email)

    # Validate that at least one of username or email is provided along with password
    if (not username and not email) or not password:
        return jsonify({"error": "Either username or email, along with password, is required!"}), 400

    # find user
    user = None
    if(username):
     user = users_collection.find_one({"username": username})
    else:
     user = users_collection.find_one({"email": email})

    if not user:
        return jsonify({"error": "User not found!"}), 404

    # Check if the password is correct
    if not check_password(user['hashed_password'], password):
        return jsonify({"error": "Incorrect password!"}), 400


    userLogin = User(user_id=user['_id'], username=user['username'])
    login_user(userLogin)  # Log the user in


    # After login, redirect the user to the original page they requested
    next_page = request.args.get('next')  # Get the 'next' parameter from the URL
    return redirect(next_page or url_for('gainer_loser'))  # Redirect to 'next' or default page




# Logout Route
@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    # session.pop('user', None)
    print( jsonify({'message': 'Logged out successfully'}))
    return redirect(url_for('gainer_loser'))










#*****************************************************************************************************************
                
@app.route('/stock_show/<stockName>/reviews', methods=['GET'])
def get_reviews(stockName):
    try:
        # Find the stock by name
        stock = db.stocks.find_one({"stockName": stockName})

        review_list = []
        if  stock:
            # return jsonify({"error": "Stock not found!"}), 404
            # Fetch all reviews associated with the stock
            reviews = db.reviews.find({"stockId": stock["_id"]})
            
            for review in reviews:
                # Optionally, get the username from the User collection
                user = db.users.find_one({"_id": review["writerId"]})
                username = user["username"] if user else "Unknown User"
                review_list.append({
                    "comment": review["comment"],
                    "writer": username,
                    "reviewId": str(review["_id"]),  
                    "writerID": str(review["writerId"]),
                    "createdAt": review["createdAt"]
                })
   
        # Render the template with reviews
        return render_template(
                                "review.html",  
                                stockName=stockName,
                                reviews=review_list
                            )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/stock_show/<stockName>/reviews', methods=['POST'])
@login_required
def post_review(stockName):
    try:

        # Check if the request content type is form-encoded
        if request.content_type == 'application/x-www-form-urlencoded':
            # Get form data
            comment = request.form.get('comment')
        else:
            return jsonify({"error": "Invalid content type, expected 'application/x-www-form-urlencoded'"}), 400


        if not comment:
            return jsonify({"error": "Comment is required!"}), 400

        # Find the stock by name
        stock = db.stocks.find_one({"stockName": stockName})

        stock_id=None
        if not stock:
            # return jsonify({"error": "Stock not found!"}), 404
            # Create a new stock document if not found
            stock_id = db.stocks.insert_one({
                "stockName": stockName,
                "reviewId_list": []
            }).inserted_id
            # stock = db.stocks.find_one({"_id": stock_id})
        else:
            stock_id=stock["_id"]

        # Create a new review object
        new_review = {
            "comment": comment,
            "createdAt":datetime.now(timezone.utc),
            "writerId": ObjectId(current_user.id),  # Use the logged-in user's ID
            "stockId": stock_id
        }

        # Insert the review into the database
        result = db.reviews.insert_one(new_review)

        # Add the review ID to the stock's review list
        db.stocks.update_one(
            {"_id": stock_id},
            {"$push": {"reviewId_list": result.inserted_id}}
        )

        print(jsonify({"message": "Review added successfully!"}), 201)
        return redirect(url_for('get_reviews', stockName=stockName))

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/stock_show/<stockName>/reviews/<reviewId>/delete_review', methods=['DELETE'])
@login_required
def stock_delete_review(stockName, reviewId):
    try:
        # Find the stock by stockName
        stock = db.stocks.find_one({"stockName": stockName})
        if not stock:
            return jsonify({"error": "Stock not found!"}), 404
        
        # Find the review by reviewId
        review = db.reviews.find_one({"_id": ObjectId(reviewId)})
        if not review:
            return jsonify({"error": "Review not found!"}), 404
        
        # Check if the current user is the one who wrote the review
        if str(review["writerId"]) != current_user.id:
            return jsonify({"error": "You are not authorized to delete this review."}), 403
        
        # Delete the review from the reviews collection
        db.reviews.delete_one({"_id": ObjectId(reviewId)})
        
        # Remove the reviewId from the stock's reviewId_list
        db.stocks.update_one(
            {"_id": stock["_id"]},
            {"$pull": {"reviewId_list": ObjectId(reviewId)}}
        )

        # Return success message
        return jsonify({"message": "Review deleted successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#*****************************************************************************************************************



def fetch_watchlist_details(watchlist):
    data = {}
    for stock in watchlist:
        try:
            data[stock]=None
            ticker = yf.Ticker(stock)
            stock_info = ticker.history(period="5d")
        
            if not stock_info.empty and len(stock_info) >= 2:

                latest_data = stock_info.iloc[-1]
                previous_data = stock_info.iloc[-2]
                percent_change = ((latest_data["Close"] - previous_data["Close"]) / previous_data["Close"]) * 100
                absolute_change = latest_data["Close"] - previous_data["Close"]
                
                data[stock] = {
                    "CLOSE_PRICE": latest_data["Close"],
                    "HIGH": latest_data["High"],
                    "LOW": latest_data["Low"],
                    "VOLUME": latest_data["Volume"],
                    "PERCENT_CHANGE": percent_change,
                    "ABSOLUTE_CHANGE": absolute_change,
                    "PREV_CLOSE":previous_data["Close"]
                }
               
        except Exception as e:
            print(e)
            raise e
            
    return  data



@app.route('/watchlist', methods=['GET'])
def watchlist():
    try:
        user_id=current_user.id
        
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            raise ValueError("User not found.")
        
        watchlist = user.get('watchlist', [])
        
        watchlist_data = fetch_watchlist_details(watchlist) if watchlist else None

        print(watchlist_data)
  
        return render_template('watchlist.html', watchlist_data=watchlist_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add_watchlist', methods=['POST'])
def add_watchlist():
    try:
        user_id = current_user.id
        stock_symbol = request.form.get('stock_symbol', '').upper()  # Ensure the symbol is in uppercase
        
        if not stock_symbol:
            raise ValueError("Stock symbol is required.")

        user = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user: 
            raise ValueError("Stock symbol is required.")
        
        watchlist = user.get('watchlist', [])
        
        if stock_symbol in watchlist:
             raise ValueError(f"{stock_symbol} is already in your watchlist")  
        
        watchlist.append(stock_symbol)
        
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"watchlist": watchlist}}
        )

        flash(f"Stock symbol {stock_symbol} added to your watchlist.", "success")
        return redirect(url_for('gainer_loser')) 

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/remove_watchlist', methods=['POST'])
def remove_watchlist():
    try:
        stock_symbol = request.form.get('stock_symbol', '').upper()  

        if not stock_symbol:
            raise ValueError("Stock symbol is required.")
        
        user = users_collection.find_one({"_id": ObjectId(current_user.id)})

        if not user: 
            raise ValueError("Stock symbol is required.")
        
        watchlist = user.get('watchlist', [])
        
        if stock_symbol in watchlist:
            watchlist.remove(stock_symbol)
            users_collection.update_one(
                {"_id": ObjectId(current_user.id)},
                {"$set": {"watchlist": watchlist}}
            )
            flash(f'{stock_symbol} removed from your watchlist', 'success')
        else:
            raise ValueError("not found in your watchlist'")
        
        # Redirect back to the watchlist page
        return redirect(url_for('watchlist'))
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)