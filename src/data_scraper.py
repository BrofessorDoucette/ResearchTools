import subprocess
import datetime
import os
from dateutil import rrule
import calendar


def download_omni(month: str, year: str, directory: str):

    subprocess.call(
        args=["wget", "--no-check-certificate", "-r", "-nd", "--no-parent", "-A", f"omni_hro2_1min_{year}{month}*.cdf",
              f"https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/{year}/"],
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=directory)


def download_year_omni(year: int, make_dirs = False, raw_data_dir ="./../raw_data/OMNI/"):

    output_dir = os.path.join(raw_data_dir, f"{year}")

    if not os.path.isdir(output_dir):
        if make_dirs:
            os.makedirs(output_dir)
        else:
            raise Exception("\nOutput directory doesn't exist, and make_dirs flag is set to false! Please make the directory or give me permission to make it for you.")

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"

        download_omni(month=_month, year=_year, directory=output_dir)
        print(f"Downloaded OMNI data for : {dt}")


def download_rept_l2(satellite: str, month: str, day: str, year: str, directory: str):

    subprocess.call(args=["wget", "--no-check-certificate", "-r", "-nd", "--no-parent", "-A",
                          f"rbsp{satellite}_rel03_ect-rept-sci-l2_{year}{month}{day}*.cdf",
                          f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite}/l2/ect/rept/sectors/rel03/{year}/"],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=directory)


def download_month_rept_l2(satellite: str, month: int, year: int,
                           make_dirs = False, raw_data_dir ="./../raw_data/REPT/"):

    if month < 10:
        output_dir = os.path.join(raw_data_dir, f"{year}/0{month}")
    else:
        output_dir = os.path.join(raw_data_dir, f"{year}/{month}")

    if not os.path.isdir(output_dir):
        if make_dirs:
            os.makedirs(output_dir)
        else:
            raise Exception("\nOutput directory doesn't exist, and make_dirs flag is set to false! Please make the directory or give me permission to make it for you.")

    start = datetime.datetime(year = year, month = month, day = 1)

    if month == 12:
        end = datetime.datetime(year = year + 1, month = 1, day = 1)
    else:
        end = datetime.datetime(year = year, month = month + 1, day = 1)

    curr = start
    while curr < end:

        _year = str(curr.year)
        _month = str(curr.month)
        _day = str(curr.day)

        if len(_month) < 2:
            _month = f"0{_month}"
        if len(_day) < 2:
            _day = f"0{_day}"

        download_rept_l2(satellite = satellite.lower(), month = _month, day = _day, year = _year, directory=output_dir)
        print(f"Downloaded REPT Data for : {curr}")
        curr += datetime.timedelta(days=1)


def download_goes_netcdf(month: str, year: str, lastday: str, directory: str):

    #https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2013GL059181

    subprocess.call(args=["wget", "--tries=0", "--retry-connrefused", "--waitretry=3", "--no-check-certificate",
                          "-r", "-nd", "--no-parent", "-A",
                          f"g15_epead_cpflux_5m_{year}{month}01_{year}{month}{lastday}.nc",
                          f"https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/{year}/{month}/goes15/netcdf/"],
                    shell=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=directory)


def download_month_goes_netcdf(month: int, year: int, make_dirs = False, raw_data_dir ="./../raw_data/GOES/"):

    output_dir = os.path.join(raw_data_dir, f"{year}/")

    if not os.path.isdir(output_dir):
        if make_dirs:
            os.makedirs(output_dir)
        else:
            raise Exception("\nOutput directory doesn't exist, and make_dirs flag is set to false! Please make the directory or give me permission to make it for you.")

    daterange = calendar.monthrange(year, month)

    _year = str(year)
    _month = str(month)

    if len(_month) < 2:
        _month = f"0{_month}"

    _lastday = str(daterange[-1])

    download_goes_netcdf(month = _month, year = _year, lastday = _lastday, directory = output_dir)
    print(f"Downloaded GOES Data for : {datetime.datetime(year=year, month=month, day=1)}")


def download_year_goes_netcdf(year: int, make_dirs = False, raw_data_dir ="./../raw_data/GOES/"):

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):

        download_month_goes_netcdf(month=dt.month, year=dt.year, make_dirs = make_dirs, raw_data_dir=raw_data_dir)


def download_poes_metop(satellite: str, month: str, day: str, year: str, directory: str):

    subprocess.call(args=["wget", "--no-check-certificate", "-r", "-nd", "--no-parent", "-A",
                          f"{satellite}_poes-sem2_fluxes-2sec_{year}{month}{day}*.cdf",
                          f"https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite}/sem2_fluxes-2sec/{year}/"],
                    shell=True, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=directory)


def download_month_poes_metop(month: int, year: int, make_dirs = False, raw_data_dir ="./../raw_data/POES_METOP/"):

    #satellites_to_try = ["metop1", "metop2", "noaa15", "noaa16", "noaa18", "noaa19"]

    satellites_to_try = ["metop1"]

    for satellite in satellites_to_try:

        if month < 10:
            raw_output_dir = os.path.join(raw_data_dir, f"{year}/0{month}/")
        else:
            raw_output_dir = os.path.join(raw_data_dir, f"{year}/{month}/")

        if not os.path.isdir(raw_output_dir):
            if make_dirs:
                os.makedirs(raw_output_dir)
            else:
                raise Exception("\nOutput directory doesn't exist, and make_dirs flag is set to false! Please make the directory or give me permission to make it for you.")

        start = datetime.datetime(year = year, month = month, day = 1)

        if month == 12:
            end = datetime.datetime(year = year + 1, month = 1, day = 1)
        else:
            end = datetime.datetime(year = year, month = month + 1, day = 1)

        curr = start
        while curr < end:

            _year = str(curr.year)
            _month = str(curr.month)
            _day = str(curr.day)

            if len(_month) < 2:
                _month = f"0{_month}"
            if len(_day) < 2:
                _day = f"0{_day}"

            download_poes_metop(satellite = satellite.lower(),
                                month = _month,
                                day = _day,
                                year = _year,
                                directory=raw_output_dir)

            print(f"Downloaded POES/METOP Data for : {curr}")
            curr += datetime.timedelta(days=1)
