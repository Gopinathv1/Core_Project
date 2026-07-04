import pandas as pd

def clean_data(data):
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Keep only numeric columns
    data = data.select_dtypes(include=['number'])  
    
    # Handle missing values (example: fill with mean)
    data = data.fillna(data.mean())
    
    # Convert categorical columns to category type
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category')
    
    return data


