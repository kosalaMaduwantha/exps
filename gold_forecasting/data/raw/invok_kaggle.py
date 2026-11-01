import kagglehub

# Download latest version
path = "data-sources/gold_price.csv"
path = kagglehub.dataset_download("tunguz/gold-prices")

print("Path to dataset files:", path)