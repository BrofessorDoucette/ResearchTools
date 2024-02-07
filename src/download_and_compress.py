import data_compressor
import data_scraper

def download_and_compress_year(satellite: str, year: int):
    
    for month in range(1, 13):
        
        data_scraper.download_month(satellite = satellite.upper(), month = month, year = year, make_dirs = True)
        data_compressor.compress_month(satellite = satellite.upper(), month = month, year = year, make_dirs = True)
        
def download_and_compress_month(satellite: str, month: int, year: int):
    
    data_scraper.download_month(satellite = satellite.upper(), month = month, year = year, make_dirs = True)
    data_compressor.compress_month(satellite = satellite.upper(), month = month, year = year, make_dirs = True)
        
download_and_compress_month(satellite="A", month=6, year=2019)
