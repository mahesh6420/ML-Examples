from data_abstract import DataAbstract


class DatasetService(DataAbstract):

    def preprocessing(self):
        #     transforming categorical features to numerical
        self.data['type'] = self.data['type'].map({
            "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3,
            "TRANSFER": 4, "DEBIT": 5
        })

