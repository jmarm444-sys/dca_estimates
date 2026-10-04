##  dca_estimates

basic dollar-cost averaging estimates utility
Enter a ticker and a run of months. The app prices a purchase on the first
Tuesday of each month in two ways — a fixed number of shares, and a fixed
dollar amount — then compares each average price per share with the current
price. Monthly rows are saved in SQLite as {ticker}_dca_{month}_{year}.db.
