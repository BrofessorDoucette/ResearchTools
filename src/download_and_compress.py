import data_compressor
import data_scraper


def download_and_compress_year_rept_l2(satellite: str, year: int):
    
    for month in range(1, 13):
        
        data_scraper.download_month_rept_l2(satellite = satellite.upper(),
                                            month = month, year = year,
                                            make_dirs = True)

        data_compressor.compress_month_rept_l2(satellite = satellite.upper(),
                                               month = month, year = year,
                                               make_dirs = True)


def download_and_compress_month_rept_l2(satellite: str, month: int, year: int):
    
    data_scraper.download_month_rept_l2(satellite = satellite.upper(),
                                        month = month, year = year,
                                        make_dirs = True)

    data_compressor.compress_month_rept_l2(satellite = satellite.upper(),
                                           month = month, year = year,
                                           make_dirs = True)
