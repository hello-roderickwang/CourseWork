CREATE DATABASE Instacart

CREATE TABLE order_products(
    order_id BIGINT,
    product_id BIGINT,
    add_to_cart_order INT,
    reordered INT
)

DROP TABLE order_products

SELECT TOP 10 * FROM order_products

SELECT TOP 10 * FROM products

CREATE TABLE MarketBasket(
    ProductA VARCHAR(100),
    ProductB VARCHAR(100),
    Frq INT
)

SELECT * FROM MarketBasket

SELECT a.product_id as p_a, b.product_id as p_b
FROM order_products a, order_products b
WHERE a.order_id = b.order_id AND a.product_id <> b.product_id

SELECT c.p_a, c.p_b, COUNT(*)
FROM(
    SELECT a.product_id as p_a, b.product_id as p_b
    FROM order_products a, order_products b
    WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
) c
GROUP BY
    c.p_a, c.p_b

INSERT INTO MarketBasket (ProductA, ProductB, Frq)
VALUES (
    SELECT c.p_a, c.p_b, COUNT(*)
    FROM(
        SELECT a.product_id as p_a, b.product_id as p_b
        FROM order_products a, order_products b
        WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
    ) c
    GROUP BY
        c.p_a, c.p_b
)

INSERT INTO MarketBasket (ProductA, ProductB, Frq)
SELECT c.p_a, c.p_b, COUNT(*)
FROM(
    SELECT a.product_id as p_a, b.product_id as p_b
    FROM order_products a, order_products b
    WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
) c
GROUP BY
    c.p_a, c.p_b

SELECT TOP 10 * FROM MarketBasket

SELECT *
FROM MarketBasket
WHERE ProductA = 13176 AND ProductB = 8106

SELECT d.p_a, d.p_b, SUM(d.frq)
FROM(
    SELECT c.p_a, c.p_b, COUNT(*) AS frq
    FROM(
        SELECT a.product_id as p_a, b.product_id as p_b
        FROM order_products a, order_products b
        WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
    ) c
    GROUP BY
        c.p_a, c.p_b
) d
WHERE d.p_a = d.p_b

DROP TABLE MarketBasket

CREATE TABLE newMarket(
    ProductA VARCHAR(100),
    ProductB VARCHAR(100),
    Frq INT
)

INSERT INTO newMarket (ProductA, ProductB, Frq)
SELECT c.p_a, c.p_b, COUNT(*)
FROM(
    SELECT a.product_id as p_a, b.product_id as p_b
    FROM order_products a, order_products b
    WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
) c
GROUP BY
    c.p_a, c.p_b

SELECT TOP 10 * FROM newMarket

SELECT * FROM newMarket WHERE ProdA = 41844 AND ProdB = 37718

SELECT * FROM newMarket WHERE ProdA = 37718 AND ProdB = 41844


INSERT INTO MarketBasket (ProductA, ProductB, Frq)
SELECT a.ProdA, a.ProdB, a.frq+b.frq
FROM newMarket a, newMarket b
WHERE a.ProdA = b.ProdB AND a.ProdB = b.ProdA
GROUP BY a.ProdA, a.ProdB, a.frq, b.frq

SELECT TOP 10 * FROM MarketBasket

SELECT * 
FROM newMarket
WHERE ProdA = 20660 and ProdB = 29987


SELECT * 
FROM MarketBasket
WHERE ProductA = 29987 and ProductB = 20660

SELECT TOP 3 m.ProductB, rec.product_name, m.Frq
FROM products p, MarketBasket m, products rec
WHERE p.product_name = 'Large Lemon' AND p.product_id = m.ProductA AND rec.product_id = m.ProductB
ORDER BY m.Frq DESC

--test UDF
SELECT * FROM dbo.RecommandProduct('Large Lemon')

DROP TABLE newMarketBasket

CREATE TABLE newMarketBasket(
    ProductA BIGINT,
    ProductB VARCHAR(MAX),
    Frq INT
)

INSERT INTO newMarketBasket (ProductA, ProductB, Frq)
SELECT a.ProdA, p.product_name, a.frq+b.frq
FROM newMarket a, newMarket b, products p
WHERE a.ProdA = b.ProdB AND a.ProdB = b.ProdA AND a.ProdB = p.product_id
GROUP BY a.ProdA, p.product_name, a.frq, b.frq

SELECT TOP 10 * FROM newMarketBasket

INSERT INTO MarketBasket (ProductA, ProductB, Frq)
SELECT c.p_a, c.p_b, COUNT(*)
FROM(
    SELECT a.product_id as p_a, b.product_id as p_b
    FROM order_products a, order_products b
    WHERE a.order_id = b.order_id AND a.product_id <> b.product_id
) c
GROUP BY
    c.p_a, c.p_b

SELECT TOP 10 * FROM MarketBasket