import matplotlib.pyplot as plt
import json

with open('./data.json', 'r') as file:
    data = json.load(file)

names = list(data.keys())
values = list(data.values())

plt.barh(names, values)
plt.xlabel('Expected Shortfall (10%)', fontsize=20)
# plt.ylabel('Method', fontsize=20)
# plt.title('Comparison of Different Approaches', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('./barchart.png', bbox_inches='tight')
