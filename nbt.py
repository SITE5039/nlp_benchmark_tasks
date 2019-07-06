import argparse

class NBT:

    def __init__(self):
        pass

    def get_cmd_parser(self):
        parser = argparse.ArgumentParser(description='NBT v0.01')
        parser.add_argument('--us_model', type = str,
                            default = 'upstream_models/test',
                            help = """location of the directory that contains
                                      upstream model.py, model.pt""")
        parser.add_argument('--task', type = str, default = 'all',
                            help = 'name of the downstream task')
        parser.add_argument('--ds_model', type=str, default='all',
                            help = """name of the downstream model for the 
                                      chosen task""")
        parser.add_argument('--ds_model_config', type = str,
                            default = 'default.json',
                            help = """location of the config file for the 
                                      chosen downstream model""")
        return parser
    
    def benchmark(self, us_model, task, ds_model, ds_model_config):
        pass


if __name__ == '__main__':

    nbt = NBT()
    parser = nbt.get_cmd_parser()
    args = parser.parse_args()
    nbt.benchmark(
        us_model = args.us_model,
        task = args.task
        ds_model = args.ds_model,
        ds_model_config = args.ds_model_config
    )