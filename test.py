import yaml

stream = open('./config.yaml', mode='r')
data = yaml.safe_load(stream)
tmp = data.get('underlier')
result = tmp.get('type')
print(result)