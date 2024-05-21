import glob
import subprocess
import datetime
import os
import os_helper
from dateutil import rrule
import calendar
import re


def wget_r_directory(folder_url: str, savdir: str) -> None:

    subprocess.call(args=["wget",
                          "--no-verbose",
                          "-e robots=off",
                          "--user-agent=Mozilla/5.0",
                          "-nc",
                          "-c",
                          "--tries=3",
                          "--retry-connrefused",
                          "--retry-on-host-error",
                          "--retry-on-http-error=500",
                          "--no-check-certificate",
                          "-r",
                          "-nd",
                          "--no-parent",
                          "-A",
                          "*.cdf",
                          folder_url,
                          "-P",
                          savdir],
                    shell=True)


def wget_file(filename: str, folder_url: str, savdir: str) -> None:
    
    subprocess.call(args=["wget",
                          "-e robots=off",
                          "--user-agent=Mozilla/5.0",
                          "--retry-connrefused",
                          "--no-check-certificate",
                          "-r",
                          "-nd",
                          "--no-parent",
                          "-A",
                          filename,
                          folder_url],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=savdir)


def download_year_omni(year: int, 
                       make_dirs: bool = False, 
                       raw_data_dir: str = "./../raw_data/OMNI/") -> None:

    output_dir = os.path.join(raw_data_dir, f"{year}")

    os_helper.verify_output_dir_exists(directory = output_dir, force_creation = make_dirs, hint="RAW OMNI DIR")

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):

        _year = str(dt.year)
        _month = str(dt.month)

        if len(_month) < 2:
            _month = f"0{_month}"

        wget_file(filename = f"omni_hro2_1min_{_year}{_month}*.cdf",
                  folder_url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro2_1min/{_year}/",
                  savdir = output_dir)
        
        print(f"Downloaded OMNI data for : {dt}")


def download_month_rept_l2(satellite: str,
                           month: int, year: int,
                           make_dirs: bool = False,
                           raw_data_dir: str = "./../raw_data/REPT/") -> None:

    if month < 10:
        output_dir = os.path.join(raw_data_dir, f"{year}/0{month}")
    else:
        output_dir = os.path.join(raw_data_dir, f"{year}/{month}")

    os_helper.verify_output_dir_exists(directory = output_dir, force_creation=make_dirs, hint="RAW REPT DIR")
    
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

        wget_file(filename = f"rbsp{satellite.lower()}_rel03_ect-rept-sci-l2_{_year}{_month}{_day}*.cdf",
                  folder_url = f"https://spdf.gsfc.nasa.gov/pub/data/rbsp/rbsp{satellite.lower()}/l2/ect/rept/sectors/rel03/{_year}/",
                  savdir = output_dir)
        
        print(f"Downloaded REPT Data for : {curr}")
        curr += datetime.timedelta(days=1)


def download_month_goes_netcdf(month: int, year: int,
                               make_dirs: bool = False,
                               raw_data_dir: str = "./../raw_data/GOES/") -> None:

    #https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2013GL059181

    output_dir = os.path.join(raw_data_dir, f"{year}/")

    os_helper.verify_output_dir_exists(directory = output_dir, force_creation = make_dirs, hint="RAW GOES DIR")

    daterange = calendar.monthrange(year, month)

    _year = str(year)
    _month = str(month)

    if len(_month) < 2:
        _month = f"0{_month}"

    _lastday = str(daterange[-1])

    wget_file(filename = f"g15_epead_cpflux_5m_{_year}{_month}01_{_year}{_month}{_lastday}.nc",
              folder_url = f"https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/{_year}/{_month}/goes15/netcdf/",
              savdir = output_dir)

    print(f"Downloaded GOES Data for : {datetime.datetime(year=year, month=month, day=1)}")


def download_year_goes_netcdf(year: int, 
                              make_dirs: bool = False, 
                              raw_data_dir: str = "./../raw_data/GOES/") -> None:

    start = datetime.datetime(year = year, month = 1, day = 1)
    end = datetime.datetime(year = year + 1, month = 1, day = 1)

    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start, until = end - datetime.timedelta(days=1))):

        download_month_goes_netcdf(month=dt.month, year=dt.year, make_dirs = make_dirs, raw_data_dir=raw_data_dir)


def download_poes(satellite: str,
                  make_dirs: bool = False,
                  raw_data_dir: str = "./../raw_data/POES/") -> None:

    output_dir = os.path.join(os.path.abspath(raw_data_dir), satellite)

    os_helper.verify_output_dir_exists(directory=output_dir,
                                       force_creation=make_dirs,
                                       hint="RAW POES DIR")

    wget_r_directory(folder_url=f"https://spdf.gsfc.nasa.gov/pub/data/noaa/{satellite.lower()}/sem2_fluxes-2sec/",
                     savdir=output_dir)


if __name__ == "__main__":

    download_poes("metop2")
