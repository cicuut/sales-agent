import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_large_dataset(num_rows=300000):
    print(f"Generating {num_rows} rows of Indonesian retail data...")

    # --- Configuration Lists ---
    
    locations = [
        'Jakarta (Grand Indonesia)', 'Jakarta (Senayan City)', 'Jakarta (Kota Kasablanka)', 
        'Jakarta (Pondok Indah Mall)', 'Surabaya (Tunjungan Plaza)', 'Surabaya (Galaxy Mall)',
        'Bandung (Paris Van Java)', 'Bandung (Trans Studio)', 'Medan (Sun Plaza)',
        'Bali (Beachwalk Kuta)', 'Yogyakarta (Ambarrukmo)', 'Semarang (Paragon City)',
        'Makassar (Trans Studio)', 'Depok (Margo City)', 'Tangerang (AEON Mall)'
    ]

    channels = [
        'In-store', 'Shopee', 'Tokopedia', 'TikTok Shop', 'Zalora', 'Website', 'Lazada'
    ]

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

    # Flatten product list for easier selection
    flat_products = []
    for cat, items in products.items():
        for name, base_price in items:
            flat_products.append({'category': cat, 'name': name, 'base_price': base_price})

    # --- Date Generation ---
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days
    
    # Generate random dates using numpy for speed
    random_days = np.random.randint(0, date_range_days, num_rows)
    dates = [start_date + timedelta(days=int(day)) for day in random_days]
    dates.sort() # Sort by date usually looks better

    # --- Data Construction ---
    data = []
    
    for date_val in dates:
        # 1. Random Selection
        prod = random.choice(flat_products)
        loc = random.choice(locations)
        channel = random.choice(channels)
        
        # 2. Price Variation (Simulate discounts/fluctuations)
        price_variance = random.uniform(0.9, 1.1)
        final_unit_price = int(prod['base_price'] * price_variance / 1000) * 1000 # Round to nearest 1000 IDR
        
        # 3. Units Sold (Weighted: lower prices sell more)
        if final_unit_price < 200000:
            units = np.random.randint(1, 15)
        elif final_unit_price < 1000000:
            units = np.random.randint(1, 6)
        else:
            units = np.random.randint(1, 3)

        revenue = units * final_unit_price

        # 4. Feature Engineering
        
        # Payday Effect: In Indonesia, usually 25th-30th or 1st-5th
        day = date_val.day
        is_payday = 1 if (day >= 25 or day <= 5) else 0

        # Holidays (Simplified Major Indo Holidays)
        is_holiday = 0
        month = date_val.month
        # New Year, Eid Al-Fitr (approx April for 2024, March for 2025), Independence (Aug), Christmas
        if (month == 1 and day == 1) or \
           (month == 8 and day == 17) or \
           (month == 12 and day == 25) or \
           (month == 4 and 9 <= day <= 12 and date_val.year == 2024) or \
           (month == 3 and 29 <= day <= 31 and date_val.year == 2025):
            is_holiday = 1

        # Promo: Higher chance on 'Twin Dates' (1.1, 2.2, etc) or Payday
        is_promo = 0
        if (month == day) or is_payday or is_holiday:
            # 60% chance of promo on special days
            if random.random() < 0.6:
                is_promo = 1
        else:
            # 10% random promo otherwise
            if random.random() < 0.1:
                is_promo = 1
        
        # Adjust Price/Sales based on Promo
        if is_promo:
            final_unit_price = int(final_unit_price * 0.8) # 20% off
            units = int(units * 1.5) # Sales boost
            revenue = units * final_unit_price

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
    columns = [
        'date', 'product_category', 'product_name', 'units_sold', 
        'unit_price', 'revenue', 'store_location', 'sales_channel', 
        'paydayeffect', 'holiday', 'promo'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    filename = 'indonesian_fashion_sales_300k.csv'
    df.to_csv(filename, index=False)
    print(f"Successfully created {filename} with {len(df)} rows.")

if __name__ == "__main__":
    generate_large_dataset(800000)