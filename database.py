from pymongo import MongoClient

# MongoDB connection
mongourl= "mongodb://127.0.0.1:27017/"
client = MongoClient(mongourl)
db = client['stock']  # Use the 'stock' database

def setup_users_collection():
    try:
        if "users" in db.list_collection_names():
            return {"message": "Users collection already exists in 'stock' database!"}

        # Define the JSON schema for the 'users' collection
        schema = {
            "bsonType": "object",
            "required": ["username", "email", "created_at", "updated_at", "hashed_password"],  # Make hashed_password required
            "properties": {
                "username": {
                    "bsonType": "string",
                    "description": "must be a string and is required"
                },
                "email": {
                    "bsonType": "string",
                    "pattern": "^.+@.+\\..+$",
                    "description": "must be a string and match the email format"
                },
                "age": {
                    "bsonType": "int",
                    "minimum": 0,
                    "description": "must be an integer and greater than or equal to 0"
                },
                "address": {
                    "bsonType": "object",
                    "properties": {
                        "street": {
                            "bsonType": "string",
                            "description": "must be a string"
                        },
                        "city": {
                            "bsonType": "string",
                            "description": "must be a string"
                        },
                        "state": {
                            "bsonType": "string",
                            "description": "must be a string"
                        },
                        "country": {
                            "bsonType": "string",
                            "description": "must be a string"
                        }
                    }
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "must be a valid ISO 8601 date and is required"
                },
                "updated_at": {
                    "bsonType": "date",
                    "description": "must be a valid ISO 8601 date and is required"
                },
                "hashed_password": {
                    "bsonType": "string",
                    "description": "must be a string and should store the hashed password"
                },
                "watchlist": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "string",
                        "description": "each item must be a string (stock name)"
                    },
                    "description": "must be an array of stock names (strings)"
                }
            }
        }

        # Create the 'users' collection with schema validation
        db.create_collection(
            "users",
            validator={"$jsonSchema": schema},
            validationLevel="strict",
            validationAction="error"
        )
        return {"message": "Users collection created successfully with validation in 'stock' database!"}
    except Exception as e:
        return {"error": str(e)}


def setup_reviews_collection():
    try:
        if "reviews" in db.list_collection_names():
            return {"message": "Reviews collection already exists in 'stock' database!"}

        # Define the JSON schema for the 'reviews' collection
        schema = {
            "bsonType": "object",
            "required": ["comment", "createdAt", "writerId"],  # Added 'stockId' to required fields
            "properties": {
                "comment": {
                    "bsonType": "string",
                    "description": "must be a string and is required"
                },
                "createdAt": {
                    "bsonType": "date",
                    "description": "must be a valid ISO 8601 date and is required"
                },
                "writerId": {
                    "bsonType": "objectId",
                    "description": "must be an ObjectId and is required, referencing the 'User' collection"
                },
                "stockId": {
                    "bsonType": "objectId",
                    "description": "must be an ObjectId and is required, referencing the 'Stock' collection"
                }
            }
        }

        # Create the 'reviews' collection with schema validation
        db.create_collection(
            "reviews",
            validator={"$jsonSchema": schema},
            validationLevel="strict",
            validationAction="error"
        )
        return {"message": "Reviews collection created successfully with validation in 'stock' database!"}
    
    except Exception as e:
        return {"error": str(e)}


def setup_stocks_collection():
    try:
        if "stocks" in db.list_collection_names():
            return {"message": "Stocks collection already exists in 'stock' database!"}

        # Define the JSON schema for the 'stocks' collection
        schema = {
            "bsonType": "object",
            "required": ["stockName"],  # Making 'stockName'  array required
            "properties": {
                "stockName": {
                    "bsonType": "string",
                    "description": "must be a string and is required"
                },
                "reviewId_list": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "objectId",
                        "description": "must be an array of ObjectIds referencing the 'reviews' collection"
                    },
                    "description": "Array of review ObjectIds"
                }
            }
        }

        # Create the 'stocks' collection with schema validation
        db.create_collection(
            "stocks",
            validator={"$jsonSchema": schema},
            validationLevel="strict",
            validationAction="error"
        )
        return {"message": "Stocks collection created successfully with validation in 'stock' database!"}
    
    except Exception as e:
        return {"error": str(e)}


# print( setup_users_collection())
# print( setup_stocks_collection())
# print( setup_reviews_collection())







# import arrow
# now_india = arrow.now('Asia/Kolkata')


# # Get the current time in India timezone (Asia/Kolkata)

# # Subtract 5 hours from the current time
# past_time_utc = now_india.shift(hours=-20000)

# # Get human-readable time difference
# time_ago = past_time_utc.humanize()
# print(now_india)  # Output: "5 hours ago"
