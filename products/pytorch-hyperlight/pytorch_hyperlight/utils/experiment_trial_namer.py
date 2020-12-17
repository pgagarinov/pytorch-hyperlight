from datetime import datetime


class ExperimentTrialNamer:
    @staticmethod
    def get_group_name():
        dt_now_string = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        return dt_now_string
