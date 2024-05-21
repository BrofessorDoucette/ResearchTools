import data_compressor
import data_scraper


def download_and_compress_year_rept_l2(satellite: str, year: int) -> None:
    
    for month in range(1, 13):
        
        data_scraper.download_month_rept_l2(satellite = satellite.upper(),
                                            month = month, year = year,
                                            make_dirs = True)

        data_compressor.compress_month_rept_l2(satellite = satellite.upper(),
                                               month = month, year = year,
                                               make_dirs = True)


def download_and_compress_month_rept_l2(satellite: str, month: int, year: int) -> None:
    
    data_scraper.download_month_rept_l2(satellite = satellite.upper(),
                                        month = month, year = year,
                                        make_dirs = True)

    data_compressor.compress_month_rept_l2(satellite = satellite.upper(),
                                           month = month, year = year,
                                           make_dirs = True)


def download_and_compress_poes_metop(satellite: str) -> None:
 
    data_scraper.download_poes(satellite=satellite.lower(),
                               make_dirs=True)

    #Retry downloading any failed files... will ignore links which overwrite existing files :)
    #Sometimes wget fails to establish an ssl connection, leading to missing files.
    data_scraper.download_poes(satellite=satellite.lower())

    data_compressor.compress_poes(satellite=satellite.lower(),
                                  make_dirs=True)


if __name__ == '__main__':

    download_and_compress_poes_metop("noaa19")
