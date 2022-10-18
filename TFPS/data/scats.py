from datetime import datetime
import os.path
import unicodecsv
import xlrd
import numpy as np
import pandas as pd


def format_date(column):
    """Formats a column of dates into the d/m/Y format with caching to increase performance"""

    dates = {date: datetime.strftime(datetime(*xlrd.xldate_as_tuple(
        date, 0)), "%d/%m/%Y") for date in column}
    return column.map(dates)


def convert_to_csv(input_name, output_name):
    """Converts the VicRoads 2006 spreadsheet into a CSV"""

    spreadsheet = xlrd.open_workbook(input_name).sheet_by_index(1)

    file = open(output_name, "wb")
    data = unicodecsv.writer(file, encoding="latin-1")

    for n in range(2, spreadsheet.nrows):
        data.writerow(spreadsheet.row_values(n))

    file.close()


class ScatsData(object):
    """ Stores all of the VicRoads 2006 data and allows for retrieval of the data"""

    DATA_SOURCE = "data/Scats Data October 2006.xls"
    CSV_FILE = "data/2006.csv"

    def __init__(self):
        if not os.path.exists(self.CSV_FILE):
            convert_to_csv(self.DATA_SOURCE, self.CSV_FILE)

        dataset = pd.read_csv(self.CSV_FILE, encoding="latin-1", sep=",", header=None)
        self.data = pd.DataFrame(dataset)
        self.data[9] = format_date(self.data[9])

    def __enter__(self):
        return self

    def get_scats_volume(self, scats_number, location):
        """Retrieves the volume at a location over the time period"""

        raw_data = self.data.loc[(self.data[0] == scats_number) & (self.data[7] == location)]

        volume_data = []
        for i in raw_data.index:
            for n in range(10, 106):
                volume_data.append(int(raw_data[n].loc[i]))

        return np.array(volume_data)

    def get_all_scats_numbers(self):
        """Gets all the scats site numbers"""

        return self.data[0].unique()

    def count(self):
        """Counts the number of rows"""

        return len(self.data.index)

    def get_location_name(self, scats_number, location):
        """Retrieves the name of the location from the VicRoads ID"""
        raw_data = self.data.loc[(self.data[0] == scats_number) & (self.data[7] == location)]
        return raw_data.iloc[0][1]

    def get_location_id(self, location_name):
        """Retrieves the location ID based on the name"""

        raw_data = self.data.loc[self.data[1] == location_name]

        return raw_data.iloc[0][7]

    def get_scats_approaches(self, scats_number):
        """Retrieves all the locations a vehicle can approach the site from"""
        raw_data = self.data.loc[self.data[0] == scats_number]
        return [int(location) for location in raw_data[7].unique()]

    def get_scats_approaches_names(self, scats_number):
        """Retrieves all the locations a vehicle can approach the site from"""
        raw_data = self.data.loc[self.data[0] == scats_number]
        return raw_data[1].unique()

    def get_positional_data(self, scats_number):
        """Retrieves the long and lat of a location"""
        raw_data = self.data.loc[(self.data[0] == scats_number)]
        return raw_data.at[raw_data.first_valid_index(),3], raw_data.at[raw_data.first_valid_index(),4]
