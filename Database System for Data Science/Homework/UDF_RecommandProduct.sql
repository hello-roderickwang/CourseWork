CREATE FUNCTION dbo.RecommandProduct (@product NVARCHAR(100))
RETURNS TABLE
AS
RETURN
(
    SELECT
        TOP 3 m.ProductB AS product_id, rec.product_name, m.Frq
    FROM
        products p, MarketBasket m, products rec
    WHERE
        p.product_name = @product AND p.product_id = m.ProductA AND rec.product_id = m.ProductB
    ORDER BY
        m.Frq DESC
)