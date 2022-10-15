from datetime import datetime
import os.path
import unicodecsv
import xlrd
import numpy as np
import pandas as pd


def format_date(column):
    dates = {date: datetime.strftime(datetime(*xlrd.xldate_as_tuple(
        date, 0)), "%d/%m/%Y") for date in column}
    return column.map(dates)


def convert_to_csv(input_name, output_name):
    spreadsheet = xlrd.open_workbook(input_name).sheet_by_index(1)

    file = open(output_name, "wb")
    data = unicodecsv.writer(file, encoding="latin-1")

    for n in range(2, spreadsheet.nrows):
        data.writerow(spreadsheet.row_values(n))

    file.close()


class ScatsData(object):
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
        raw_data = self.data.loc[(self.data[0] == scats_number) & (self.data[7] == location)]

        volume_data = []
        for i in raw_data.index:
            for n in range(10, 106):
                volume_data.append(int(raw_data[n].loc[i]))

        return np.array(volume_data)

    def get_all_scats_numbers(self):
        return self.data[0].unique()

    def count(self):
        return len(self.data.index)

    def get_location_name(self, scats_number, location):
        raw_data = self.data.loc[(self.data[0] == scats_number) & (self.data[7] == location)]

        return raw_data.iloc[0][1]

    def get_location_id(self, location_name):
        raw_data = self.data.loc[self.data[1] == location_name]

        return raw_data.iloc[0][7]

    def get_scats_approaches(self, scats_number):
        raw_data = self.data.loc[self.data[0] == scats_number]

        return [int(location) for location in raw_data[7].unique()]

    def get_positional_data(self, scats_number, location):
        raw_data = self.data.loc[(self.data[0] == scats_number) & (self.data[7] == location)]

        return raw_data[3].loc[0], raw_data[4].loc[0]
