from nsetools-master import nse

nse = Nse()
# Get all stock codes
stock_codes = nse.get_stock_codes()
print(stock_codes)
