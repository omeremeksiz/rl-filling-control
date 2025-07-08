CREATE TABLE IF NOT EXISTS parsed_filling_data_table (
    id SERIAL PRIMARY KEY,
    raw_data TEXT NOT NULL,
    coarse_time FLOAT,
    fine_time FLOAT,
    total_time FLOAT,
    switching_state INT,
    overflow_amount FLOAT,
    underflow_amount FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
