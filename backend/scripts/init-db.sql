-- Database Initialization Script
-- This script runs automatically when PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search optimization

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE crypto_prediction_db TO crypto_user;

-- Create custom types (if needed in future)
-- Example: CREATE TYPE price_trend AS ENUM ('up', 'down', 'stable');

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully at %', NOW();
END $$;
