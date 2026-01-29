# Import required libraries for data generation and manipulation
import pandas as pd  # Data manipulation and DataFrame creation
import numpy as np  # Numerical operations and random number generation
import random  # Random selection from lists
from datetime import datetime, timedelta  # Date and time operations

# Function to generate a large synthetic dataset of Indonesian retail sales data
def generate_large_dataset(num_rows=300000):
    print(f"Generating {num_rows} rows of Indonesian retail data...")

    # --- Configuration Lists ---
    # List of major shopping mall locations across Indonesia
    locations = [
        'Jakarta (Grand Indonesia)', 'Jakarta (Senayan City)', 'Jakarta (Kota Kasablanka)', 
        'Jakarta (Pondok Indah Mall)', 'Surabaya (Tunjungan Plaza)', 'Surabaya (Galaxy Mall)',
        'Bandung (Paris Van Java)', 'Bandung (Trans Studio)', 'Medan (Sun Plaza)',
        'Bali (Beachwalk Kuta)', 'Yogyakarta (Ambarrukmo)', 'Semarang (Paragon City)',
        'Makassar (Trans Studio)', 'Depok (Margo City)', 'Tangerang (AEON Mall)'
    ]

    # List of sales channels (both physical and online platforms)
    channels = [
        'In-store', 'Shopee', 'Tokopedia', 'TikTok Shop', 'Zalora', 'Website', 'Lazada'
    ]

    # Dictionary of product categories with items and their base prices in Indonesian Rupiah (IDR)
    products = {
        "Women's Clothing": [
            ('Batik Maxi Dress', 450000), ('Kebaya Modern', 1200000), ('Denim Jacket', 499000),
            ('Pleated Skirt', 225000), ('Cotton Blouse', 180000), ('Tunik Muslimah', 250000)
        ],
        "Men's Clothing": [
            ('Batik Shirt Long Sleeve', 550000), ('Slim Fit Chinos', 350000), 
            ('Graphic T-Shirt', 120000), ('Tailored Suit Jacket', 2500000), ('Koko Shirt', 200000)
        ],
        "Footwear": [
            ('Leather Pantofel', 850000), ('Running Sneakers', 1200000), 
            ('Slip-on Loafers', 650000), ('Canvas Sneakers', 250000), ('Platform Sandals', 350000)
        ],
        "Accessories": [
            ('Silk Hijab', 125000), ('Leather Belt', 150000), ('Sling Bag', 185000),
            ('Gold Plated Necklace', 250000), ('Aviator Sunglasses', 150000)
        ],
        "Activewear": [
            ('Yoga Leggings', 250000), ('Performance Hoodie', 450000), 
            ('Sports Bra', 199000), ('Running Shorts', 180000), ('Jersey Bola', 150000)
        ]
    }

    # Flatten the product dictionary into a single list for easier random selection
    flat_products = []
    for cat, items in products.items():
        for name, base_price in items:
            flat_products.append({'category': cat, 'name': name, 'base_price': base_price})

    # --- Date Generation ---
    # Define the date range for the synthetic dataset (2024 and 2025)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    # Calculate total number of days in the range
    date_range_days = (end_date - start_date).days
    
    # Generate random day offsets within the date range using numpy for performance
    random_days = np.random.randint(0, date_range_days, num_rows)
    # Convert day offsets to actual dates
    dates = [start_date + timedelta(days=int(day)) for day in random_days]
    # Sort dates chronologically for better data presentation
    dates.sort()

    # --- Data Construction ---
    # Initialize list to store all generated transaction records
    data = []
    
    # Iterate through each date and generate transaction data
    for date_val in dates:
        # 1. Random Selection - Pick random product, location, and sales channel
        prod = random.choice(flat_products)
        loc = random.choice(locations)
        channel = random.choice(channels)
        
        # 2. Price Variation - Simulate price fluctuations and discounts
        # Apply a random variance between 90% and 110% of base price
        price_variance = random.uniform(0.9, 1.1)
        # Round final price to nearest 1000 IDR for realistic pricing
        final_unit_price = int(prod['base_price'] * price_variance / 1000) * 1000
        
        # 3. Units Sold - Weighted model where lower prices sell more units (demand elasticity)
        if final_unit_price < 200000:
            # Budget items: higher unit sales (1-14 units)
            units = np.random.randint(1, 15)
        elif final_unit_price < 1000000:
            # Mid-range items: moderate unit sales (1-5 units)
            units = np.random.randint(1, 6)
        else:
            # Premium items: lower unit sales (1-2 units)
            units = np.random.randint(1, 3)

        # Calculate total revenue for the transaction
        revenue = units * final_unit_price

        # 4. Feature Engineering - Create additional features that influence sales
        
        # Payday Effect: In Indonesia, salary days are typically around the 25th-30th or 1st-5th
        # This flag indicates if the transaction occurred on a payday
        day = date_val.day
        is_payday = 1 if (day >= 25 or day <= 5) else 0

        # Holiday Detection - Flag major Indonesian holidays that boost sales
        is_holiday = 0
        month = date_val.month
        # Check for major holidays: New Year, Eid Al-Fitr, Independence Day, Christmas
        if (month == 1 and day == 1) or \
           (month == 8 and day == 17) or \
           (month == 12 and day == 25) or \
           (month == 4 and 9 <= day <= 12 and date_val.year == 2024) or \
           (month == 3 and 29 <= day <= 31 and date_val.year == 2025):
            is_holiday = 1

        # Promo Flag - Determine if a promotional discount is applied
        # Higher chance of promos on special dates (payday, holidays, or "lucky" dates like 1.1, 2.2)
        is_promo = 0
        if (month == day) or is_payday or is_holiday:
            # 60% chance of promo on these special occasions
            if random.random() < 0.6:
                is_promo = 1
        else:
            # 10% chance of random promos on regular days
            if random.random() < 0.1:
                is_promo = 1
        
        # Adjust Price and Units based on Promo Status
        if is_promo:
            # Apply 20% discount when promo is active
            final_unit_price = int(final_unit_price * 0.8)
            # Increase units sold by 50% due to promotional effect
            units = int(units * 1.5)
            # Recalculate revenue with adjusted price and units
            revenue = units * final_unit_price

        # Add the complete transaction record to the data list
        data.append([
            date_val.strftime('%Y-%m-%d'),
            prod['category'],
            prod['name'],
            units,
            final_unit_price,
            revenue,
            loc,
            channel,
            is_payday,
            is_holiday,
            is_promo
        ])

    # --- Create DataFrame and Save ---
    # Define the column names for the dataset
    columns = [
        'date', 'product_category', 'product_name', 'units_sold', 
        'unit_price', 'revenue', 'store_location', 'sales_channel', 
        'paydayeffect', 'holiday', 'promo'
    ]
    
    # Create a pandas DataFrame from the generated data
    df = pd.DataFrame(data, columns=columns)
    
    # Write the DataFrame to a CSV file for storage and analysis
    filename = 'indonesian_fashion_sales_300k.csv'
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename} with {len(df)} rows.")

# Entry point: Execute the dataset generation function when script is run directly
if __name__ == "__main__":
    generate_large_dataset(800000)