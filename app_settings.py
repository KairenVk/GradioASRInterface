import yaml


with open('settings.yaml', 'r', encoding='utf8') as file:
    settings_yaml = yaml.load(file, Loader=yaml.FullLoader)
