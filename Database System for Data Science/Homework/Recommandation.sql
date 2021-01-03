-- Create MarketBasket Table
CREATE TABLE MarketBasket(
    ProductA VARCHAR(100),
    ProductB VARCHAR(100),
    Frq INT
)

-- Check structure of MarketBasket Table
SELECT * FROM MarketBasket

-- Fill in MarketBasket Table
INSERT INTO MarketBasket (ProductA, ProductB, Frq)
SELECT c.p_a, c.p_b, COUNT(*)
FROM(
    SELECT a.product_id as p_a, b.product_id as p_b
    FROM order_products a, order_products b
    WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
) c
GROUP BY
    c.p_a, c.p_b

-- Check data for MarketBasket Table
SELECT TOP 10 * FROM MarketBasket

--test UDF
SELECT * FROM dbo.RecommandProduct('Limes')

--test UDF
SELECT * FROM dbo.RecommandProduct('Large Lemon')