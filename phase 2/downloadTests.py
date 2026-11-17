import duckdb

url = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/fhvhv_tripdata_2024-04.parquet"
)

keep = {
    "hvfhs_license_num",
    "request_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "pulocationid",
    "dolocationid",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "driver_pay",
    "tips",
}

con = duckdb.connect()
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

full_size = con.execute("""
    SELECT file_size_bytes
    FROM parquet_file_metadata(?)
""", [url]).fetchone()[0]

columns = con.execute("""
    SELECT
        path_in_schema,
        SUM(total_compressed_size) AS bytes
    FROM parquet_metadata(?)
    GROUP BY 1
    ORDER BY bytes DESC
""", [url]).fetchall()

selected_size = 0

for name, size in columns:
    if name.lower() in keep:
        selected_size += size

print("Original:", round(full_size / 1e6, 2), "MB")
print("Selected:", round(selected_size / 1e6, 2), "MB")
print("Reduction:", round(
    100 * (1 - selected_size / full_size), 2
), "%")