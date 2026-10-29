
## Main independent variable will be predicted
### GOLD Data

**URL**: https://eco3min.fr/en/gold-price-history-dataset/

Above URL provides the LBMA gold price from 1960 to the present time 
** information of the data set can be obtained from the above source
https://eco3min.fr/dataset/gold-price.csv is the download link
frequency: monthly

second dataset 
from kaggle
location in the project: data-sources/gold

### Kitco daily gold (daily model target)

**File**: `data-sources/gold_data_new.xlsx`
**Used by**: `model_building/gold_daily_prophet.ipynb`
**Frequency**: daily business-day panel; missing weekdays are not fabricated
**Target**: `Kitco Price (USD)`
**Date field**: `Dates` only; the separate `Date` column is ignored
**Coverage after cleaning**: 2015-02-05 → 2026-10-26
**URL**: not recorded in this repository; verify the original Kitco source before redistribution

### WGC monthly gold (legacy monthly model only)

**File**: `data-sources/Gold_price_averages_in_a range_of_currencies_since_1978.xlsx`
**Used by**: `model_building/prophet.ipynb`
**Frequency**: monthly
**Note**: no longer used as a daily QA cross-check.

### US interest rates

**URL**: https://fred.stlouisfed.org/data/FEDFUNDS

Above URL provides the US interest rates from the FRED data source
please refer to the above link for more details about the data set
frequency: monthly
path: D:\gold_prediction\data-sources\US-interest-rates-use.csv

### US 10Y Treasury Yield

**URL**: https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data
frequency: daily

### US Dollar Index (DXY)

**URL**: https://www.investing.com/indices/usdollar-historical-data
frequency: monthly in the downloaded file (despite the site offering daily data); the daily model upsamples it with a release lag

### US CPI Inflation
taken from FRED data source
Need further assistance 

### WTI Crude Oil

**URL**: https://markets.ft.com/data/commodities/tearsheet/historical?c=WTI+Crude+Oil
Above source contains the WTI Crude oil data need scraping to extract data
**Daily model file**: `data-sources/wti_historical.csv` (daily `Close`, newest-first weekday-prefixed dates)

### Brent Crude Oil

**URL**: https://markets.ft.com/data/commodities/tearsheet/historical?c=Brent+Crude+Oil
Above source contains the Brent Crude oil data need scraping to extract data
**Daily model file**: `data-sources/brent_historical.csv` (daily `Close`, newest-first weekday-prefixed dates)

### Legacy FRED crude oil inputs

`DCOILWTICO.csv` / `DCOILBRENTEU.csv` are retained for the legacy monthly notebook only; the daily notebook uses the historical CSVs above.





