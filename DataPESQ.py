<<<<<<< HEAD
from xlrd import open_workbook
from cachetools import cached, TTLCache
cache = TTLCache(maxsize=10000, ttl=10000)

class DataPESQ:
    def __init__(self, path, column_list=['PESQ', 'Lossrate', 'Real mbls']):
        self.wb = open_workbook(path)
        self.sheet = self.wb.sheets()[0]
        self.nr_of_rows = self.sheet.nrows
        self.nr_of_columns = self.sheet.ncols
        self.columns_indexes = []
        row = self.sheet.row(0)
        row_values = list(map(lambda x: x.value, row))
        for name in column_list:
            self.columns_indexes.append(row_values.index(name))

    @cached(cache)
    def get_data(self):
        input_data = []
        for i in range(1,self.nr_of_rows):
            row = []
            for j in self.columns_indexes:
                row.append(self.sheet.cell(i,j).value)
            input_data.append(row)
        return input_data
=======
from xlrd import open_workbook
from cachetools import cached, TTLCache
cache = TTLCache(maxsize=10000, ttl=10000)

class DataPESQ:
    def __init__(self, path, column_list=['PESQ', 'Lossrate', 'Real mbls']):
        self.wb = open_workbook(path)
        self.sheet = self.wb.sheets()[0]
        self.nr_of_rows = self.sheet.nrows
        self.nr_of_columns = self.sheet.ncols
        self.columns_indexes = []
        row = self.sheet.row(0)
        row_values = list(map(lambda x: x.value, row))
        for name in column_list:
            self.columns_indexes.append(row_values.index(name))

    @cached(cache)
    def get_data(self):
        input_data = []
        for i in range(1,self.nr_of_rows):
            row = []
            for j in self.columns_indexes:
                row.append(self.sheet.cell(i,j).value)
            input_data.append(row)
        return input_data
>>>>>>> d9ac9156e476000b8d9f7ed97955090d3cee208e
