import yaml
class Reader:


    CONFIG_DIR_SERVER = "./Config/service_config.yaml"
    CONFIG_DIR_AI_SERVER = "./Config/server_config.yaml"
    
    def __init__(self):
        pass
    def get_config_server(self):
         
        with open(self.CONFIG_DIR_SERVER) as file:
            # Load config data
            data = yaml.load(file, Loader=yaml.FullLoader)

            # Close file
            file.close()
        return data
    
    
    def get_config_AI_server(self):
        with open(self.CONFIG_DIR_AI_SERVER) as file:
            # Load config data
            data = yaml.load(file, Loader=yaml.FullLoader)

            # Close file
            file.close()
        return 
